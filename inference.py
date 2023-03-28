from ultralytics import YOLO

PROJECT = 'experiments/contrastive'
EPOCHS = 50
CONTRASTIVE_LOSS = 0.05
WORKERS = 24
EXPERIMENT = 'isaid_contrastive' + str(CONTRASTIVE_LOSS) + 'epochs_' + str(EPOCHS)

model = YOLO("runs/detect/isaid_contrastive0.9epochs_502/weights/last.pt")
model.train(data="isaid.yaml", epochs=EPOCHS, batch=8, save_period=9, workers=12, resume='runs/detect/isaid_contrastive0.9epochs_502/weights/last.pt', name=EXPERIMENT, contr_loss=CONTRASTIVE_LOSS)
