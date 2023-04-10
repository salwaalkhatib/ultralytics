# Ultralytics YOLO 🚀, GPL-3.0 license
from copy import copy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel


# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):

    def get_dataloader(self, dataset_path, batch_size, mode="train", rank=0):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 augment=mode == "train",
                                 cache=self.args.cache,
                                 pad=0 if mode == "train" else 0.5,
                                 rect=self.args.rect or mode == "val",
                                 rank=rank,
                                 workers=self.args.workers,
                                 close_mosaic=self.args.close_mosaic != 0,
                                 prefix=colorstr(f'{mode}: '),
                                 shuffle=mode == "train",
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, rank=rank, mode=mode,
                             rect=mode == "val", names=self.data['names'])[0]

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        # nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'Contr_loss'
        return v8.detect.DetectionValidator(self.test_loader,
                                            save_dir=self.save_dir,
                                            logger=self.console,
                                            args=copy(self.args))

    def criterion(self, preds, contr_feats, batch):
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = Loss(de_parallel(self.model), contr_feats, epoch=self.epoch, 
                                     contr_warmup=self.contr_warmup, contr_pnorm=self.contr_pnorm, 
                                     contr_ema_iters=self.contr_ema_iters, queue_size=self.queue_size)
            self.compute_loss.iters = 0
        self.compute_loss.epoch = self.epoch
        return self.compute_loss(preds, contr_feats, batch)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ('\n' + '%9s' * (4 + len(self.loss_names)) + 
                '%9s' * len(self.model.names)) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size', *[k[:8] for k in self.model.names.values()])

    def plot_training_samples(self, batch, ni):
        plot_images(images=batch["img"],
                    batch_idx=batch["batch_idx"],
                    cls=batch["cls"].squeeze(-1),
                    bboxes=batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"train_batch{ni}.jpg")

    def plot_metrics(self):
        plot_results(file=self.csv)  # save results.png


# Criterion class for computing training losses
class Loss:

    def __init__(self, model, contr_feats, epoch, contr_warmup, contr_pnorm, contr_ema_iters, queue_size=10): # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.prototypes = Prototypes(self.nc, contr_feats, queue_size=queue_size)
        self.epoch = epoch
        self.contr_warmup = contr_warmup
        self.contr_pnorm = contr_pnorm
        self.contr_ema_iters = contr_ema_iters

        self.use_dfl = m.reg_max > 1
        roll_out_thr = h.min_memory if h.min_memory > 1 else 64 if h.min_memory else 0  # 64 is default

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))      # Returns (B,8400,4), self.proj is a tensor of vals from 0-16 ie: np.arange(16). 16 is some DFL parameter
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def contrastive_loss_stage(self, contrastive_features, targets, fg_mask):
        '''
        Compute contrastive loss between features and centroids
        contrasitve_features: list of tensors from each detection stage (3) 
        '''
        # Indexes to split the three stages in targets
        feat_st1, feat_st2, feat_st3 = contrastive_features
        st1_sz, st2_sz, st3_sz = feat_st1.shape[-1] * feat_st1.shape[-2], feat_st2.shape[-1] * feat_st2.shape[-2], feat_st3.shape[-1] * feat_st3.shape[-2]
        st1, st2, st3 = st1_sz, st1_sz + st2_sz, st1_sz + st2_sz + st3_sz
        targ_st1, targ_st2, targ_st3 = targets[:, :st1], targets[:, st1:st2], targets[:, st2:st3]
        fg_mask1, fg_mask2, fg_mask3 = fg_mask[:, :st1], fg_mask[:, st1:st2], fg_mask[:, st2:st3]
        centroid_st1, centroid_st2, centroid_st3 = self.prototypes.get_centroids()
        loss_input = zip([feat_st1, feat_st2, feat_st3], [targ_st1, targ_st2, targ_st3], [fg_mask1, fg_mask2, fg_mask3], [centroid_st1, centroid_st2, centroid_st3])
        
        loss_contr = 0
        for feat, targ, mask, centroid in loss_input:
            # Compute loss for each stage
            anchors = feat.view(feat.shape[0], feat.shape[1], -1)
            anchors = anchors.transpose(1, -1)
            loss_contr += self.contrastive_loss(anchors[mask], centroid, targ[mask]) if targ[mask].sum() > 0 else 0
        return loss_contr
    
    def contrastive_loss(self, anchors, centroids, targets, margin=1):
        device = anchors.device
        centroids = centroids.to(device)
        anchor_distances = torch.cdist(anchors.float(), centroids, p=self.contr_pnorm) # L1/L2 distance
        gt_labels = torch.zeros_like(anchor_distances, device=device) - 1
        gt_labels[torch.arange(gt_labels.shape[0]), targets.ravel().type(torch.int64)] = 1
        loss = F.hinge_embedding_loss(anchor_distances, gt_labels, margin=margin)
        
        return loss.mean()


    def __call__(self, preds, contr_features, batch):
        self.iters += 1
        targets_seen = torch.zeros(self.nc, device=self.device)
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, Contrastive
        feats = preds[1] if isinstance(preds, tuple) else preds
        # Concat from P3(80x80=6400), P4(40x40=1600), P5(20x20=400) and split into (16x4, num_classes)
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)    # Stupid, Hardcode #Salwa

        # targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1) # (num_instances, 6) batchID, classID, 4 bbox coords
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])    # (B, max_num_instances, 5) Max num instances for each batch, rest is zero
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) # Mask appended zeros (checks which predictions are just bunch of zeros)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Why is this taking a pred_scores with Sigmoid not softmax?
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)   #(B,8400) labels, (B, 8400, 4) target boxes, (B,8400,80) target class scores(this is not one-hot encoded), mask(B, 8400)
        # fg_mask tells which feature maps are predicting an object

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():   # Checking if there are predictions assigned
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        # Keep Track of number of target for each class
        if fg_mask.sum() > 0:
            t, c = batch['cls'].unique(return_counts=True)
            targets_seen[t.type(torch.long)] += c.to(self.device)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # Contrastive loss
        ## self.prototypes.update_centroids(contr_features, target_labels)
        ## loss_contrastive = self.contrastive_loss_stage(contr_features, target_labels)
        if (self.epoch > self.contr_warmup//2) and (self.epoch < self.contr_warmup):
            if (self.iters % self.contr_ema_iters == 0) and (not self.validate):
                self.prototypes.update_centroids(contr_features, target_labels, fg_mask)
        elif self.epoch >= self.contr_warmup:
            if fg_mask.sum():
                loss[3] = self.contrastive_loss_stage(contr_features, target_labels, fg_mask)
                if (self.iters % self.contr_ema_iters == 0) and (not self.validate):
                    self.prototypes.update_centroids(contr_features, target_labels, fg_mask)
        
        loss[3] *= self.hyp.contr_loss

        return loss.sum() * batch_size, loss.detach(), targets_seen  # loss(box, cls, dfl, contrastive)


class Prototypes():
    def __init__(self, num_classes, contr_feats, queue_size=10):
        self.device = contr_feats[0].device
        self.queue_size = queue_size
        self.num_classes = num_classes
        self.calibrator = PrototypeRecalibrator(num_classes=num_classes)
        self.momentum = 0.9
        self.centroids = [torch.zeros(num_classes, feat.shape[1]).to(self.device) for feat in contr_feats]
        # self.centroids = [torch.rand(num_classes, feat.shape[1]) for feat in contr_feats]
        self.queues = [torch.zeros(self.queue_size, num_classes, feat.shape[1]).to(self.device) for feat in contr_feats]

    @torch.no_grad()
    def update_centroids(self, features, targets, fg_mask):
        '''
        Updates the centroids of the clusters
        '''
        # Get targets for each stage of prediction
        st1_sz, st2_sz, st3_sz = features[0].shape[-1] * features[0].shape[-2], features[1].shape[-1] * features[1].shape[-2], features[2].shape[-1] * features[2].shape[-2]
        st1, st2, st3 = st1_sz, st1_sz + st2_sz, st1_sz + st2_sz + st3_sz
        targ_labels = [targets[:, :st1], targets[:, st1:st2], targets[:, st2:st3]]
        fg_masks = [fg_mask[:, :st1], fg_mask[:, st1:st2], fg_mask[:, st2:st3]]

        # Update queue
        for i in range(len(self.centroids)):
            if fg_masks[i].sum() == 0:
                continue
            feats = features[i].view(features[i].shape[0], features[i].shape[1], -1)
            feats = feats.transpose(1, -1)  # (N, 64/128/256)
            anc_feats = feats[fg_masks[i].bool()]  # Extract only assigned features using mask
            targ = targ_labels[i][fg_masks[i].bool()]  # Extract corresponding labels
            queue = self.queues[i]
            for cl in range(15):
                cl_feats = anc_feats[targ==cl]
                num_feats = cl_feats.shape[0]
                if num_feats > self.queue_size:
                    # queue[:, cl, :] = torch.cat((cl_feats[:self.queue_size-1, :], cl_feats[self.queue_size:, :].mean(0).unsqueeze(0)), 0)
                    queue[:, cl, :] = cl_feats[:self.queue_size, :]
                elif (num_feats > 0) and (num_feats <=  self.queue_size):
                    queue[:, cl, :] = torch.cat((cl_feats[:num_feats, :], queue[num_feats:, cl, :]), 0)

        # Update centroids
        for i in range(len(self.centroids)):
            self.centroids[i] = self.momentum * self.centroids[i] + (1 - self.momentum) * self.queues[i].mean(0)
            # self.calibrator.update(self.centroids[i], features[i], targ_labels[i], fg_masks[i])
            # self.centroids[i] = self.calibrator.recalibrate(self.centroids[i])


    def get_centroids(self):
        '''
        Returns the centroids of the clusters
        '''
        return self.centroids

class PrototypeRecalibrator():
    def __init__(self, beta=0.95, initial_wc=0.01, num_classes=15):
        self.beta = beta # smoothing coefficient
        self.wc = [initial_wc for _ in range(num_classes)]
        self.num_classes = num_classes
    
    def update(self, prototypes, features, targets, fg_masks):
        # update based on a batch of data
        # use an exponential moving average

        feats = features.view(features.shape[0], features.shape[1], -1)
        feats = feats.transpose(1, -1)
        feats = feats[fg_masks.bool()]
        targets = targets[fg_masks.bool()]
        for cl in range(self.num_classes):
            feat_cl = feats[targets==cl]
            prot_cl = prototypes[cl]
            N = feat_cl.shape[0]
            if N == 0:
                continue
            exps = 1 / (1 + torch.exp(-1 * torch.matmul(feat_cl, prot_cl.unsqueeze(-1).type(torch.float16))))
            wc_batch = torch.sum(exps) / N
            self.wc[cl] = (self.beta * self.wc[cl] + (1 - self.beta) * wc_batch).item()
    
    def recalibrate(self, prototypes):
        # recalibrate prototypes
        new_prototypes = prototypes.clone()
        for i in range(self.num_classes):
            new_prototypes[i] = prototypes[i] + torch.log(torch.tensor(self.wc[i]))
        return new_prototypes

def train(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or "yolov8n.pt"
    data = cfg.data or "coco128.yaml"  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()


if __name__ == "__main__":
    train()
