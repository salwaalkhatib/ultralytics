from ultralytics import YOLO

PROJECT = 'runs/detect/v8s_hinge-1'
EPOCHS = 70
CONTRASTIVE_LOSS = 0.1
WORKERS = 24
MOSAIC_PROB = 1
CLOSE_MOSAIC = 10
EXPERIMENT = 'isaid_contr_momen_noCalib' + str(CONTRASTIVE_LOSS) + '_queue10_' + str(EPOCHS) + 'epochs' + '_mosaic' + str(MOSAIC_PROB) + '_closemosaic' + str(CLOSE_MOSAIC)
CKPT = 'yolov8s.pt'

model = YOLO(CKPT)
model.train(data="isaid.yaml", epochs=EPOCHS, batch=8, save_period=9, workers=WORKERS, name=EXPERIMENT, project=PROJECT, contr_loss=CONTRASTIVE_LOSS, mosaic=MOSAIC_PROB, close_mosaic=CLOSE_MOSAIC)
