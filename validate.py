from ultralytics import YOLO

PROJECT = 'runs/detect/vals'
EPOCHS = 70
CONTRASTIVE_LOSS = 0.2
WORKERS = 16
EXPERIMENT = 'isaid_contrastive' + str(CONTRASTIVE_LOSS) + '_queue10_' + str(EPOCHS) + 'epochs'
CKPT = 'runs/detect/v8s-ContrNew-hinge2/shortlist/isaid_contr_momen_Calib_0.2queue50_emaIters200_70epochs_mosaic1.0_closemosaic10/weights/best.pt'
print(CKPT)

model = YOLO(CKPT)
metrics = model.val()  # evaluate model performance on the validation set
# results = model(source="data/val/images/", save_json=True, save_txt=True, save_conf=True)  # run inference on the validation set