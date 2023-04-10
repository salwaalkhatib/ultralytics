from ultralytics import YOLO

PROJECT = 'runs/detect/vals'
EPOCHS = 70
CONTRASTIVE_LOSS = 0.1
WORKERS = 32
EXPERIMENT = 'isaid_contrastive' + str(CONTRASTIVE_LOSS) + '_queue10_' + str(EPOCHS) + 'epochs'
CKPT = 'runs/detect/v8s-hingeL2/isaid_contr_momen_NoCalib_0.05queue10_emaIters100_70epochs_mosaic1.0_closemosaic10/weights/best.pt'

model = YOLO(CKPT)
metrics = model.val(save_json=True)  # evaluate model performance on the validation set
# results = model(source="data/val/images/", save_json=True, save_txt=True, save_conf=True)  # run inference on the validation set