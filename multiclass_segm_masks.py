import cv2
import numpy as np
import os

def coco_masks(coco_data, masks_dir):
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    colors = {
        1: (255, 0, 0),
        2: (0, 255, 0)
    }

    for image_info in coco_data['images']:
        mask = np.zeros((image_info['height'], image_info['width'], 3), dtype=np.uint8)

        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]

        for ann in annotations:
            class_id = ann['category_id']
            color = colors.get(class_id, (0, 0, 0))

            for segmentation in ann['segmentation']:
                polygon = np.array(segmentation, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [polygon], color)

        mask_path = os.path.join(masks_dir, f"{os.path.splitext(image_info['file_name'])[0]}_mask.tif")          
        cv2.imwrite(mask_path, mask)

        print(f"mask saved for {image_info['file_name']}")


import json 
import glob

def process_coco_files(coco_dir, masks_dir):
    json_files = glob.glob(os.path.join(coco_dir, "*.json"))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        coco_masks(coco_data, masks_dir)


coco_dir = "path to coco json files"
masks_dir = "path to save masks"

process_coco_files(coco_dir, masks_dir)
