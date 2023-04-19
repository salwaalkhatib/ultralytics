from ultralytics import YOLO
import os

PROJECT = 'runs/detect/vals'
EPOCHS = 70
CONTRASTIVE_LOSS = 0.2
WORKERS = 16
CKPT = 'runs/detect/v8s-ContrNew-hinge2/shortlist/isaid_contr_momen_Calib_0.2queue50_emaIters100_70epochs_mosaic1.0_closemosaic5/weights/best.pt'
EXP_NAME = os.path.split(os.path.split(CKPT)[0])[0].split('/')[-1]
print(CKPT)

model = YOLO(CKPT)
metrics = model.val(save_json=True, project=PROJECT, conf=0.2, name=EXP_NAME+'NMS0.2')  # evaluate model performance on the validation set
# results = model(source="data/val/images/", save_json=True, save_txt=True, save_conf=True)  # run inference on the validation set