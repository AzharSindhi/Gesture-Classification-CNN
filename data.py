from tkinter import image_names
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import os
import cv2
import utils
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt


class GestureDataset(Dataset):
    def __init__(
        self,
        base_path,
        context_path,
        image_shape,
        num_classes,
        mode="train",
        include_raw_keypoints=True,
        include_distances=True,
        include_context=True,
        include_skelton=True,
    ):
        super().__init__()
        self.mode = mode
        self.input_shape = image_shape
        # self.data = self.downsample_class(data, 0, 0.6)
        self.context_path = context_path
        self.crops_dir = os.path.join(base_path, 'crops')
        self.keypoints_dir = os.path.join(base_path, 'keypoints')
        self.skelton_path = os.path.join(base_path, 'skelton')
        self.selected_features = [
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
        self.selected_keypoint_indices = [0, 9, 10, 5, 6, 7, 8]
        self.keypoints_mean = -7.998891e-09
        self.keypoints_std = 0.9636241
        self.distances_mean = 6.102380298432849e-08
        self.distances_std = 0.9759000205956285
        self.include_raw_keypoints = include_raw_keypoints
        self.include_distances = include_distances
        self.include_context = include_context
        self.include_skelton = include_skelton
        self.in_channels = 3
        self.keypoints_length = 21
        self.distance_features_length = 21
        self.in_channels = 3
        if self.include_skelton:
            self.in_channels += 3
        if self.include_context:
            self.in_channels += 3

        # output = self.process_and_read_data()
        # self.image_paths = output[0]
        # self.features = output[1]
        # self.bboxes = output[2]
        # self.labels = output[3]
        class_counts = [
            len(os.listdir(os.path.join(self.crops_dir, str(x)))) for x in [0, 1]
        ]
        self.crops_paths = glob.glob(
            os.path.join(self.crops_dir, "*", "*.jpg"))

        # sanity
        self.crops_paths = []
        neg_paths = glob.glob(os.path.join(self.crops_dir, "0", "*.jpg"))
        pos_paths = glob.glob(os.path.join(self.crops_dir, "1", "*.jpg"))
        np.random.shuffle(neg_paths)
        np.random.shuffle(pos_paths)
        self.crops_paths.extend(neg_paths[:500])
        self.crops_paths.extend(pos_paths)
        class_counts = [500, len(pos_paths)]

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]

        train_transforms = A.ReplayCompose(
            [
                # A.SmallestMaxSize(max_size=160),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.2
                ),
                # A.Perspective(p=0.3),
                # A.Transpose(p = 1.0),
                # A.RandomCrop(height=self.input_shape[0], width=self.input_shape[1], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                           b_shift_limit=15, p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False),
        )

        val_transforms = train_transforms
        self._img_transform = {
            "train": train_transforms, "val": val_transforms}

        self.num_classes = num_classes  # len(np.unique(self.labels))
        # class_indices, class_counts = np.unique(self.labels, return_counts=True)
        pos_class_weight = class_counts[0] / class_counts[1]
        if self.num_classes > 1:
            self.class_weights = [1, pos_class_weight]
        else:
            self.class_weights = [pos_class_weight]

        print("INFO({}): Total counts of classes:{}".format(
            self.mode, class_counts))
        print("INFO({}): Class weights:".format(self.mode), self.class_weights)

    def calculate_pos_weights(class_counts,total_len):
        pos_weights = np.ones_like(class_counts)
        neg_counts = [total_len-pos_count for pos_count in class_counts]
        for cdx, (pos_count, neg_count) in enumerate(zip(class_counts,  neg_counts)):
            pos_weights[cdx] = neg_count / (pos_count + 1e-5)

        return torch.as_tensor(pos_weights, dtype=torch.float)

    def downsample_class(self, df, class_id, factor):
        minority_df = df[df["label"] != class_id]
        majority_df = df[df["label"] == class_id]
        # num_samples = int(len(filtered_df["labels"]) * factor)
        downsampled_df = majority_df.sample(frac=1 - factor, random_state=42)
        return pd.concat([downsampled_df, minority_df])

    def process_and_read_data(self):
        image_paths = []
        features = []
        labels = []
        bboxes = []
        for idx, row in self.data.iterrows():
            img_name = row["image_name"]
            row_features = row[self.selected_features].values
            bbox = row["bbox"]
            label = int(row["label"])
            image_path = os.path.join(self.images_path, img_name)

            image_paths.append(image_path)
            features.append(row_features)
            bboxes.append(bbox)
            labels.append(label)

        return image_paths, features, bboxes, labels

    def visualize(self, orig_img, crop_img, skelton_img, keypoints, outpath='out.jpg'):
        cropped_copy = crop_img.copy()
        cropped_copy = utils.draw_keypoints(cropped_copy, keypoints)
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(orig_img)
        axes[0, 1].imshow(crop_img)
        axes[1, 0].imshow(cropped_copy)
        axes[1, 1].imshow(skelton_img)
        fig.tight_layout()
        plt.savefig(outpath)

    def apply_transformations(self, transforms, img, keypoints, replay=None):
        if replay == None:
            transformed_data = transforms(
                image=img, keypoints=keypoints
            )
        else:
            transformed_data = A.ReplayCompose.replay(
                replay, image=img, keypoints=keypoints)

        cropped_img_orig = transformed_data["image"]
        cropped_keypoints_rescaled = np.array(transformed_data["keypoints"])
        return cropped_img_orig, cropped_keypoints_rescaled, transformed_data['replay']

    def __len__(self):
        # print("Data length: ", len(self.data))
        return len(self.crops_paths)

    def __getitem__(self, index):
        # load crop image
        crop_path = self.crops_paths[index]
        cropped_img_orig = cv2.imread(crop_path)
        label = np.array(int(crop_path.split("/")[-2]))
        crop_img_name = crop_path.split("/")[-1]
        name_split = crop_img_name.split('__')
        # read full (context) image
        context_img_name = name_split[0]
        context_img = cv2.imread(os.path.join(
            self.context_path, context_img_name))
        # load keypoints
        keypoint_name = name_split[0] + '__' + name_split[1].split('.')[0]
        keypoints_path = os.path.join(
            self.keypoints_dir, str(label), str(keypoint_name) + ".npy"
        )
        cropped_keypoints = np.load(keypoints_path)
        keypoints_confidences = cropped_keypoints[:, -1].flatten()
        cropped_keypoints = cropped_keypoints[:, :2]
        cropped_keypoints[cropped_keypoints < 0] = 0

        # resize image and transform keypoints
        cropped_img_resized = cv2.resize(cropped_img_orig, self.input_shape)
        context_img_resized = cv2.resize(context_img, self.input_shape)
        cropped_keypoints_rescaled = utils.rescale_keypoints(
            cropped_keypoints, cropped_img_orig.shape, self.input_shape, order='hw')

        # read skeleton image
        skelton_path = os.path.join(
            self.skelton_path, str(label), crop_img_name)
        skelton_image = np.zeros_like(cropped_img_resized)
        transforms = self._img_transform[self.mode]
        if transforms:
            cropped_img_resized, cropped_keypoints_rescaled, replay = self.apply_transformations(
                transforms, cropped_img_resized, cropped_keypoints_rescaled)
            context_img_resized, _, _ = self.apply_transformations(
                transforms, context_img_resized, cropped_keypoints_rescaled, replay)
            skelton_image = utils.draw_keypoints(skelton_image, cropped_keypoints_rescaled)
            skelton_image, _, _ = self.apply_transformations(transforms, skelton_image, cropped_keypoints_rescaled, replay)


        distance_features = utils.calculate_1D_distances_nose(
            cropped_keypoints_rescaled)

        if self.num_classes > 1:
            label = torch.tensor(label)
            label = torch.nn.functional.one_hot(label, self.num_classes).float()

        features = torch.tensor(distance_features).float()
        #cropped_img_resized = torch.transpose(cropped_img_resized, (2, 0, 1))
        #skelton_image = torch.transpose(skelton_image, (2, 0, 1))
        image_concat = cropped_img_resized  # .copy()
        if self.include_skelton:
            image_concat = torch.cat((cropped_img_resized, skelton_image), dim=0)
        
        if self.include_context:
            image_concat = torch.cat((image_concat, context_img_resized), dim=0)
        # normalizing cropped_keypoints
        cropped_keypoints_rescaled = torch.from_numpy(
            cropped_keypoints_rescaled).float()
        cropped_keypoints_rescaled = (cropped_keypoints_rescaled - self.keypoints_mean) / self.keypoints_std
        # normalize distance features
        features = (features - self.distances_mean)/self.distances_std
        # add confidence to keypoints
        keypoints_confidences = keypoints_confidences[:, None]
        confidences = torch.from_numpy(keypoints_confidences)
        cropped_keypoints_rescaled = torch.cat(
            (cropped_keypoints_rescaled, confidences), dim=1).flatten()
        
        dist_keypoints = cropped_keypoints_rescaled if self.include_raw_keypoints else torch.zeros(0)
        if self.include_distances:
            dist_keypoints = torch.cat((dist_keypoints, features))
        
        return image_concat, dist_keypoints, label


if __name__ == "__main__":
    # /net/cluster/azhar/mywork/datasets/task_smellpersons_personbboxes/unique_images/"
    base_dir = '/net/cluster/azhar/olfactory-gestures-azhar/Gesture Classification CNN/data/data_context_train_filtered'
    context_path = "/net/cluster/azhar/mywork/datasets/task_smellpersons_personbboxes/all_images"
    input_size = (224, 224)
    features_shape = (28, 28)
    batch_size = 1
    train_dataset = GestureDataset(
        base_dir, context_path, input_size, "val")
    # train_dataset.transform = False
    val_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    out_path = "./out"
    i = 0
    for img, kpt_distances, label in val_dataloader:
        # if i == 10:
        #     break

        img, kpt_distances, label = img[0], kpt_distances[0], label[0]
        cropped_img_resized = img[:3, :, :].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        skelton_img_resized = img[3:6, :, :].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        context_img_resized = img[6:, :, :].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        keypoints = kpt_distances[:21].cpu().numpy().reshape(-1, 3)[:, :2]
        distances = kpt_distances[21:].cpu().numpy().tolist()

        # print(keypoints.shape, distances.shape)
        # print(label)
        outpath = os.path.join(out_path, str(i) + ".jpg")
        train_dataset.visualize(context_img_resized, cropped_img_resized, skelton_img_resized, keypoints, outpath)
        if i > 50:
            break
        i+=1
    # print("keypoints mean and std", np.mean(keypoints_all), np.std(keypoints_all))
    # print("distances mean and std", np.mean(distances), np.std(distances))