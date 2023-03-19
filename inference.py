from ultralytics import YOLO

PROJECT = 'experiments/init'
EPOCHS = 50
EXPERIMENT = 'small_ep' + str(EPOCHS)
CONTRASTIVE_LOSS = 0.9

model = YOLO("yolov8n.pt")
model.train(data="coco128.yaml", epochs=2, save_period=9, name=EXPERIMENT, contr_loss=CONTRASTIVE_LOSS)