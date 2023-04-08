from ultralytics import YOLO

PROJECT = 'runs/detect/test'
EPOCHS = 70
CONTRASTIVE_LOSS = 0.1
WORKERS = 32
MOSAIC_PROB = 1
CLOSE_MOSAIC = 10
EXPERIMENT = 'isaid_contr_momen_Calib' + str(CONTRASTIVE_LOSS) + 'queue10_' + str(EPOCHS) + 'epochs' + '_mosaic' + str(MOSAIC_PROB) + '_closemosaic' + str(CLOSE_MOSAIC)
# CKPT = 'yolov8s.pt'
CKPT = 'runs/detect/v8s_hinge-1/isaid_contr_momen_Calib0.1_queue10_70epochs_mosaic1_closemosaic10/weights/last.pt'

model = YOLO(CKPT)
model.train(data="isaid.yaml", epochs=EPOCHS, batch=8, save_period=9, workers=WORKERS, resume=CKPT, name=EXPERIMENT, project=PROJECT, contr_loss=CONTRASTIVE_LOSS, mosaic=MOSAIC_PROB, close_mosaic=CLOSE_MOSAIC)
