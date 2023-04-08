from ultralytics import YOLO

PROJECT = 'runs/detect/v8s'
EPOCHS = 70
CONTRASTIVE_LOSS = 0.1
WORKERS = 32
EXPERIMENT = 'isaid_contrastive' + str(CONTRASTIVE_LOSS) + '_queue10_' + str(EPOCHS) + 'epochs'
CKPT = 'runs/detect/v8s_hinge-1/isaid_contr_momen_Calib0.1_queue10_70epochs_mosaic1_closemosaic10/weights/last.pt'

model = YOLO(CKPT)
metrics = model.val(save_json=True)  # evaluate model performance on the validation set
# results = model(source="data/val/images/", save_json=True, save_txt=True, save_conf=True)  # run inference on the validation set