from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
    vis_pose_result,
)

from mmdet.apis import inference_detector, init_detector
from mmpose.datasets import DatasetInfo
import warnings
import numpy as np


class PoseDetectionModel:
    def __init__(self):
        # "/net/cluster/azhar/Keypoints-Classification/configs/faster_rcnn_r50_fpn_coco.py"  # "/configs/faster_rcnn_r50_fpn_coco.py"
        # "/configs/faster_rcnn_r50_fpn_coco.py"
        self.det_config = "./keypoints_det/faster_rcnn_r50_fpn_50e_deart_person.py"
        # "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        self.det_checkpoint = "/net/cluster/azhar/mmdetection-ODOR/work_dirs/deart/train_lme200/latest.pth"
        self.pose_config = "/net/cluster/azhar/Keypoints-Classification/configs/pose_configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
        self.pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
        self.bbox_thr = 0.6
        self.kpt_thr = 0.3
        self.det_cat_id = 1
        self.device = "cuda:0"
        self.selected_keypoints_names = [
            "nose",  # 0
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",  # 5
            "right_shoulder",  # 6
            "left_elbow",  # 7
            "right_elbow",  # 8
            "left_wrist",  # 9
            "right_wrist",  # 10
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        self.selected_keypoint_indices = [0, 9, 10, 5, 6, 7, 8]

        self.det_model = init_detector(
            self.det_config, self.det_checkpoint, device=self.device.lower()
        )
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(
            self.pose_config, self.pose_checkpoint, device=self.device.lower()
        )
        self.dataset = self.pose_model.cfg.data["test"]["type"]
        self.dataset_info = self.pose_model.cfg.data["test"].get(
            "dataset_info", None)

        if self.dataset_info is None:
            warnings.warn(
                "Please set `dataset_info` in the config."
                "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
                DeprecationWarning,
            )
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)

    def process_bboxes(self, person_dets):
        for person in person_dets:
            x1, y1, w, h = person['bbox']
            x2 = x1 + w
            y2 = y1 + h
            bbox = [x1, y1, x2, y2, 1.0]
            person["bbox"].append(1.0)#= bbox
        
        return person_dets

    def predict(self, image_path, person_results=None):
        if person_results == None:
            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(self.det_model, image_path)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(
                mmdet_results, self.det_cat_id)

        # test a single image, with a list of bboxes.
        else:
            person_results = self.process_bboxes(person_results)

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            image_path,
            person_results,
            bbox_thr=self.bbox_thr,
            format="xyxy",
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names,
        )
        # print('visualizing image')
        # img = vis_pose_result(
        #     self.pose_model,
        #     image_path,
        #     pose_results,
        #     dataset=self.dataset,
        #     dataset_info=self.dataset_info,
        #     kpt_score_thr=self.kpt_thr,
        #     radius=4,
        #     thickness=1,
        #     show=False,
        #     out_file="./out.jpg")

        return pose_results
