from ultralytics import YOLO

PROJECT = 'runs/detect/v8s'
EPOCHS = 70
CONTRASTIVE_LOSS = 0.1
WORKERS = 24
EXPERIMENT = 'isaid_contrastive' + str(CONTRASTIVE_LOSS) + '_queue10_' + str(EPOCHS) + 'epochs'

model = YOLO("runs/detect/v8s/isaid_contrastive0.1_queue10_70epochs/weights/best.pt")
metrics = model.val(save_json=True)  # evaluate model performance on the validation set
# results = model(source="data/val/images/", save_json=True, save_txt=True, save_conf=True)  # run inference on the validation set