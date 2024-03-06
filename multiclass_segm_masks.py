import cv2
import numpy as np
import os

def coco_masks(coco_data, masks_dir):
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    for image_info in coco_data['images']:
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]

        for ann in annotations:
            class_id = ann['category_id']

            for segmentation in ann['segmentation']:
                polygon = np.array(segmentation, np.int32)
                polygon = polygon.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [polygon], class_id)

        mask_path = os.path.join(masks_dir, f"{image_info['file_name']}_mask.png")          
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


coco_dir = "coco"
masks_dir = "masks"

process_coco_files(coco_dir, masks_dir)
