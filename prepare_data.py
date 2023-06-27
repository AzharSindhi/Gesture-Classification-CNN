import cv2
import os
import numpy as np
import utils
import torch
from tqdm import tqdm
import torch_model as models
import matplotlib.pyplot as plt
import albumentations as A
from keypoints_det.pose_det_model import PoseDetectionModel
import json
import pandas as pd
import glob
import torch.nn.functional as F
import glob
from pycocotools.coco import COCO

images_path = (
    "/net/cluster/azhar/mywork/datasets/task_smellpersons_personbboxes/all_images/"
)

json_path = "/net/cluster/azhar/mywork/datasets/task_smellpersons_personbboxes/annotations/test.json"
selected_keypoints = [0, 9, 10, 5, 6, 7, 8]

keypoints_detector = PoseDetectionModel()
save_path_images = "/net/cluster/azhar/olfactory-gestures-azhar/Gesture Classification CNN/data/data_context_test/images"
save_path_keypoints = "/net/cluster/azhar/olfactory-gestures-azhar/Gesture Classification CNN/data/data_context_test/keypoints"
skelton_path = '/net/cluster/azhar/olfactory-gestures-azhar/Gesture Classification CNN/data/data_context_test/skelton'
coco = COCO(json_path)
img_keys = list(coco.imgs.keys())


def get_cropped_image(img, bbox):
    x1, y1, x2, y2 = bbox
    # x2 = x1 + w
    # y2 = y1 + h

    if y1 >= img.shape[0]:
        y1 = y1 - 20
    if x1 >= img.shape[0]:
        x1 = x1 - 20

    offset = 10
    cropped_image = img[y1:y2 + offset, x1:x2 + offset]
    return cropped_image


def get_categories(coco):
    # Category IDs.
    cat_ids = coco.getCatIds()
    # All categories.
    cats = coco.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    return cat_names


categories = ["none", "holding the nose", "sniffing"]
print("categories:", categories)
for cat in categories:
    os.makedirs(os.path.join(save_path_images, cat), exist_ok=True)
    os.makedirs(os.path.join(save_path_keypoints, cat), exist_ok=True)
    os.makedirs(os.path.join(skelton_path, cat), exist_ok=True)

for i in tqdm(range(len(img_keys))):
    # get bounding box annotations
    image_id = img_keys[i]
    image = coco.loadImgs(image_id)[0]
    image_path = os.path.join(images_path, image['file_name'])
    # if not os.path.isfile(image_path):
    #     continue
    image_name = image["file_name"]
    ann_ids = coco.getAnnIds(image_id)

    # make person bounding boxes
    person_results = []
    for ann_id in ann_ids:
        person = {}
        ann = coco.anns[ann_id]
        # bbox format is 'xywh'
        x1, y1, w, h = ann["bbox"]
        x2 = x1 + w
        y2 = y1 + h
        person['bbox'] = [x1, y1, x2, y2]#ann['bbox']
        person['category'] = ann['attributes']["smell gesture"]
        person_results.append(person)
    image = cv2.imread(image_path)
    #pos_result = keypoints_detector.predict(img, person_results)
    for i, result in enumerate(person_results):
        img = image.copy()
        bbox_result =np.array(result["bbox"]).astype(float)
        category = result["category"]
        box_conf = 1.0 #bbox_result[-1]
        bbox = bbox_result[:4].astype(int)
        cropped_image = get_cropped_image(img, bbox)
        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            print("rejecting:", cropped_image.shape, bbox)
            continue
        
        result['bbox'] = [0, 0, cropped_image.shape[1], cropped_image.shape[0]]
        kpt_result = keypoints_detector.predict(cropped_image, [result])[0]
        keypoints = kpt_result["keypoints"][selected_keypoints, :3]
        keypoints_vis = kpt_result["keypoints"][selected_keypoints, :2].astype(int)

        skelton_image = np.zeros_like(cropped_image)
        skelton_image = utils.draw_keypoints(
            skelton_image, keypoints_vis)

        # nose_conf = result["keypoints"][0, -1]
        # max_wrist_conf = max(result["keypoints"][[9, 10], -1])


        cropped_name = image_name + "__" + str(i) + ".jpg"
        skelton_name = image_name + "__" + str(i) + ".jpg"
        keypoints_name = image_name + "__" + str(i) + ".npy"

        cv2.imwrite(os.path.join(skelton_path, category,
                    skelton_name), skelton_image)
        cv2.imwrite(os.path.join(save_path_images, category,
                    cropped_name), cropped_image)
        np.save(os.path.join(save_path_keypoints, category,
                             keypoints_name), keypoints)
