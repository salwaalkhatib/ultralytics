from ultralytics import YOLO

PROJECT = 'init'
EPOCHS = 50
EXPERIMENT = 'small_ep' + str(EPOCHS)

model = YOLO("yolov8s.pt")
model.train(data="coco128.yaml", epochs=EPOCHS, save_period=9, batch=8, project=PROJECT, name=EXPERIMENT)