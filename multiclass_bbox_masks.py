import cv2
import numpy as np
import os

def coco_masks(coco_data, masks_dir):
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    for image_info in coco_data['images']:
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        #mask = np.zeros((3888, 5184), dtype=np.uint8)

        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]

        for ann in annotations:
            x, y, w, h = map(int, ann['bbox'])
            class_id = ann['category_id']
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        mask_filename = os.path.join(masks_dir, f"{image_info['file_name']}_mask.png")
        cv2.imwrite(mask_filename, mask)

        print(f"mask saved for {image_info['file_name']}")


import json
import glob

def process_coco_files(coco_dir, masks_dir):
    json_files = glob.glob(os.path.join(coco_dir, '*.json'))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        coco_masks(coco_data, masks_dir)


coco_dir = "path to coco json files"
masks_dir = "path to save masks"


process_coco_files(coco_dir, masks_dir)