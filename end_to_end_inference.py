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

images_path = (
    "/net/cluster/azhar/mywork/datasets/odeuropa_46k/imgs"
)
# images_path = (
#     "data/images_split_cleaned/val/"
# )
classifier_checkpoint_path = (
    "checkpoints/d053ecdbcf5d413999c52c2a13d24338_resnet18_skelton_cleaned_data.ckp"
)
labels_path = "/net/cluster/azhar/mywork/detrex/coco-predictions/odeuropa_51k/odeuropa_51k_all_0.05_new.json"
include_linear_distances = True
include_skelton = True
# metadata = pd.read_csv(metadata_path)
# gesture_df = metadata[metadata["Search Term"].isin(
#     ["holding the nose", "sniffing"])]
# gesture_filenames = np.unique(gesture_df["File Name"].values)
# image_names = os.listdir(images_path)[:100]
num_classes = 2
selected_keypoints = [0, 9, 10, 5, 6, 7, 8]
input_shape = (224, 224)
in_channels = 6
vis_path = "./data/overlayed_temp"
i = 0
transforms = A.Compose(
    [
        # A.SmallestMaxSize(max_size=160),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
        #                    rotate_limit=15, p=0.5),
        # A.RandomCrop(height=input_shape[0], width=input_shape[1]),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15,
        #            b_shift_limit=15, p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)
# transforms = None


def save_json(path, images, categories, annotations, last_idx=None):
    coco_json = {
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    with open(path, "w") as f:
        f.write(json.dumps(coco_json, ensure_ascii=True))
    # with open(path, "w") as f:
    #     json.dump(coco_json, f)
    #     # json.dump(r, f)
    # # sanity check
    with open(path, "r") as f:
        data = json.loads(f.read())
        print(data.keys())

    if last_idx != None:
        with open("last_index.txt", "w") as f:
            f.write(str(last_idx))


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval()
    model = model.cuda()
    return model


gesture_classifier = models.CustomModel(
    num_classes, in_channels=in_channels, use_linear_branch=include_linear_distances
)
gesture_model = load_model(gesture_classifier, classifier_checkpoint_path)
keypoints_detector = PoseDetectionModel()


def prepare_inference_input(
    cropped_img_orig, cropped_keypoints, distance_features, include_keypoints=False, include_skelton=False
):
    if transforms:
        transformed_data = transforms(
            image=cropped_img_orig, keypoints=cropped_keypoints
        )
        cropped_img_orig = transformed_data["image"]
        cropped_keypoints = np.array(transformed_data["keypoints"])

    skelton_image = np.zeros(cropped_img_orig.shape, dtype=np.uint8)
    skelton_image = utils.draw_keypoints(skelton_image, cropped_keypoints)

    features = torch.tensor(distance_features).float()
    cropped_img_orig = np.transpose(cropped_img_orig, (2, 0, 1))
    skelton_image = np.transpose(skelton_image, (2, 0, 1))
    if include_skelton:
        image_concat = np.concatenate((cropped_img_orig, skelton_image), 0)
    else:
        image_concat = cropped_img_orig
    image_concat = image_concat[np.newaxis, :, :, :]
    image_concat = torch.from_numpy(image_concat).float()
    features = features.view(1, len(features))
    return image_concat, features


def get_model_prediction(X):
    img_x, keypoints_x = X
    img_x = img_x.cuda()
    keypoints_x = keypoints_x.cuda()
    pred = F.softmax(gesture_model(img_x, keypoints_x))
    pred = pred.detach().cpu().numpy()[0]
    class_idx = int(pred.argmax())
    conf = float(pred[class_idx])
    return class_idx, conf


holding_nose_id = 101
with open(labels_path, "r") as f:
    data = json.load(f)
images_paths = glob.glob(os.path.join(
    images_path, "*.jpg"))  #
# images_paths = data["images"]
# np.random.shuffle(images_paths)
images = []
img_id = 0
categories = [
    {
        "id": 0,
        "name": "background"

    },
    {
        "id": 1,
        "name": "holding nose"
    }]

annotations = []  # data["annotations"]

# with open("gesture_predictions_test.json", "r") as f:
#     data = json.load(f)
#     images = data["images"]
#     annotations = data["annotations"]

ann_id = 0  # len(annotations) + 1
save_path = "output_jsons/gestures_predictions_odeuropa51k.json"
last_img_id = 0  # max([img["id"] for img in images])

for i, image_dict in enumerate(tqdm(images_paths[:2000])):
    # image_path = image_dict
    if i <= last_img_id:
        continue
    if isinstance(image_dict, str):
        image_path = image_dict
    else:
        image_name = image_dict["file_name"]
        image_path = os.path.join(images_path, image_name)

    # image_path.split("/")[-2] + "/" + image_path.split("/")[-1]
    filename_full = image_path.split("/")[-1]
    print(filename_full)
    # filename, ext = os.path.splitext(filename_full)
    # filename_modified = filename[:20] + ext
    # filename_modified = str(filename_modified)

    if not os.path.isfile(image_path):
        print(image_path)
        continue
    img = cv2.imread(image_path)
    try:
        pos_result = keypoints_detector.predict(img)
    except Exception as e:
        print(e)
        continue
    img_dict = {
        "id": int(i),
        "file_name": filename_full,
        "height": int(img.shape[0]),
        "width": int(img.shape[1]),
        "license": None,
        "coco_url": None
    }
    images.append(img_dict)
    for result in pos_result:
        print(result.keys())
        bbox_result = result["bbox"].astype(float)
        box_conf = bbox_result[-1]
        bbox = bbox_result[:4]
        keypoints = result["keypoints"][selected_keypoints, :2].astype(int)
        nose_conf = result["keypoints"][0, -1]
        max_wrist_conf = max(result["keypoints"][[9, 10], -1])
        # if nose_conf < 0.7 or max_wrist_conf < 0.7:
        #     continue
        # print(nose_conf, max_wrist_conf)
        cropped_img_orig, cropped_keypoints = utils.resize_image(
            img, bbox, keypoints, input_shape
        )
        distance_features = utils.calculate_1D_distances_nose(
            cropped_keypoints)
        X = prepare_inference_input(
            cropped_img_orig, cropped_keypoints, distance_features, include_skelton=include_skelton)
        # print(X[0].shape, X[1].shape)
        class_idx, conf = get_model_prediction(X)
        # class_name = class_names[class_idx]
        if class_idx == 0:
            class_id = 0
        else:
            class_id = 1  # holding_nose_id

        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        color = (0, 0, 255)
        keypoints = result["keypoints"][selected_keypoints, :3]
        keypoints[:, -1] = 1
        keypoints = keypoints.flatten().tolist()
        if np.isnan(conf):
            conf = 0
        # print(conf)
        keypoints = [int(x) for x in keypoints]
        # print(keypoints)
        ann = {
            "id": int(ann_id),
            "image_id": int(img_dict["id"]),
            "category_id": class_id,
            "bbox": [int(x1), int(y1), int(w), int(h)],
            "area": float(w*h),
            "score": float(conf),
            "keypoints": keypoints
        }
        # print(json.dumps(ann))
        ann_id += 1
        annotations.append(ann)

    if (i+1) % 1000 == 0:
        save_json(save_path, images, categories,
                  annotations, last_idx=ann_id - 1)

save_json(save_path, images, categories, annotations)
