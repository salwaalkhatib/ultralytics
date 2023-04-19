import json
import os

pred_json_path = 'runs/detect/vals/contr_momen_Calib_0.2queue50_emaIters200_70epochs_mosaic1.0_closemosaic10/predictions.json'
json_save_path = os.path.split(pred_json_path)[0] + '/predictions_updated.json'
gt_json_path = 'data/val/instancesonly_filtered_val.json'

with open(pred_json_path, 'r') as f:
    pred_json = json.load(f)

with open(gt_json_path, 'r') as f:
    gt_json = json.load(f)

i = 0
for pred in pred_json:
    img_name = pred['image_id']
    for gt in gt_json['images']:
        if os.path.splitext(gt['file_name'])[0] == img_name:
            pred['image_id'] = gt['id']
            i += 1
            break

print("Number of predictions: ", len(pred_json))
print("Number of predictions updated: ", i)
with open(json_save_path, 'w') as f:
    json.dump(pred_json, f)
