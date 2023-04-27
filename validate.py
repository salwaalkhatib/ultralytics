from ultralytics import YOLO
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO args')
    parser.add_argument('--checkpoint', type=str, default='runs/detect/v8s-BCE/trainVal_contr_momen_SingleStage_BS10_noValCalib_0.2queue50_emaIters200_70epochs_mosaic1.0_closemosaic52/weights/best.pt', help='YOLO model checkpoint')
    args = parser.parse_args()

    EPOCHS = 70
    CONTRASTIVE_LOSS = 0.2
    WORKERS = 16
    CKPT = args.checkpoint
    EXP_NAME = os.path.split(os.path.split(CKPT)[0])[0].split('/')[-1]
    print(CKPT)

    model = YOLO(CKPT)

    PROJECT = 'runs/detect/valsApr27'
    metrics = model.val(save_json=True, project=PROJECT, name=EXP_NAME)  # evaluate model performance on the validation set