from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO args')
    parser.add_argument('--project', type=str, default='runs/detect/v8s-hingeL', help='path to dataset root')
    parser.add_argument('--experiment', type=str, default='isaid_contr_momen_', help='Experiment name head')
    parser.add_argument('--checkpoint', type=str, default='yolov8s.pt', help='YOLO model checkpoint')
    parser.add_argument('--epochs', default=70, type=int, help='Epochs')
    parser.add_argument('--contr_loss', default=0.1, type=float, help='Weightage for contrastive loss')
    parser.add_argument('--mosaic_prob', default=1, type=float, help='Mosaic probability')
    parser.add_argument('--close_mosaic', default=10, type=int, help='Epochs to close mosaic')
    parser.add_argument('--contr_pnorm', default=1, type=int, help='p-norm of contrastive loss')
    parser.add_argument('--contr_warmup_epochs', default=10, type=int, help='Epochs towarmup before contrastive loss')
    parser.add_argument('--queue_size', default=10, type=int, help='Queue size')
    parser.add_argument('--contr_ema_iters', default=100, type=int, help='Iterations before centroid ema iters')
    parser.add_argument('--contr_calib', action='store_true', default=False, help="Use calibration in Contrastive loss")

    args = parser.parse_args()

    WORKERS = 32
    EPOCHS = args.epochs
    CKPT = args.checkpoint
    CONTR_WARMUP = args.contr_warmup_epochs
    CONTR_EMA_ITERS = args.contr_ema_iters
    QUEUE_SIZE = args.queue_size

    CONTR_CALIB = args.contr_calib
    CONTRASTIVE_PNORM = args.contr_pnorm
    CONTRASTIVE_LOSS = args.contr_loss
    MOSAIC_PROB = args.mosaic_prob
    CLOSE_MOSAIC = args.close_mosaic
    PROJECT = args.project + str(CONTRASTIVE_PNORM)
    args.experiment = args.experiment + 'Calib' if CONTR_CALIB else args.experiment + 'NoCalib'
    EXPERIMENT = args.experiment + '_' + str(CONTRASTIVE_LOSS) + 'queue' + str(QUEUE_SIZE) + '_' + 'emaIters' + str(CONTR_EMA_ITERS) + '_' + str(EPOCHS) + 'epochs' + '_mosaic' + str(MOSAIC_PROB) + '_closemosaic' + str(CLOSE_MOSAIC)
    
    model = YOLO(CKPT)
    model.train(data="isaid.yaml", epochs=EPOCHS, batch=8, save_period=9, workers=WORKERS, name=EXPERIMENT, 
                project=PROJECT, contr_loss=CONTRASTIVE_LOSS, mosaic=MOSAIC_PROB, close_mosaic=CLOSE_MOSAIC,
                contr_pnorm=CONTRASTIVE_PNORM, contr_warmup=CONTR_WARMUP, contr_ema_iters=CONTR_EMA_ITERS, queue_size=QUEUE_SIZE, contr_calib=CONTR_CALIB)
