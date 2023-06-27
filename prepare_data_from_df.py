import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import str2list, get_cropped_image
import cv2
import utils

df_path = '/net/cluster/azhar/Keypoints-Classification/data/postprocessed_df_N-H_min_area_filtered_area_filter_merged.csv'
image_dir = '/net/cluster/azhar/mywork/datasets/task_smellpersons_personbboxes/all_images'
df = pd.read_csv(df_path)
save_dir = '/net/cluster/azhar/olfactory-gestures-azhar/Gesture Classification CNN/data/data_context_all/'
crops_path = os.path.join(save_dir, 'crops')
keypoints_path = os.path.join(save_dir, 'keypoints')
skelton_path = os.path.join(save_dir, 'skelton')
selected_keypoints = [0, 9, 10, 5, 6, 7, 8]
selected_features = [
    "nose_x",
    "nose_y",
    "left_wrist_x",
    "left_wrist_y",
    "right_wrist_x",
    "right_wrist_y",
    "left_shoulder_x",
    "left_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_y",
    "left_elbow_x",
    "left_elbow_y",
    "right_elbow_x",
    "right_elbow_y",
]
for c in [0, 1]:
    os.makedirs(os.path.join(crops_path, str(c)), exist_ok=True)
    os.makedirs(os.path.join(keypoints_path, str(c)), exist_ok=True)
    os.makedirs(os.path.join(skelton_path, str(c)), exist_ok=True)

for i, row in tqdm(df.iterrows()):
    img_path = os.path.join(image_dir, row['image_name'])
    img = cv2.imread(img_path)
    image_name = row['image_name'] + '__' + str(i)
    bbox = np.array(str2list(row['bbox']), dtype=float)
    bbox = bbox[:4].astype(int)
    category = str(row['label'])
    # keypoints = np.array(
    #     str2list(row['keypoints']), dtype=float).reshape(-1, 3)
    # keypoints = keypoints[selected_keypoints, :2]
    keypoints = row[selected_features].values.reshape(-1, 2).astype(int)
    cropped_image = get_cropped_image(img, bbox)
    if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
        print("rejecting:", cropped_image.shape)
        continue

    skelton_image = np.zeros(cropped_image.shape, dtype=np.uint8)
    skelton_image = utils.draw_keypoints(
        cropped_image, keypoints)
    cv2.imwrite(os.path.join(skelton_path, category,
                image_name + '.jpg'), skelton_image)
    cv2.imwrite(os.path.join(crops_path, category,
                image_name + '.jpg'), cropped_image)
    # save the skelteon
    np.save(os.path.join(keypoints_path, category,
            image_name + '.npy'), keypoints)
