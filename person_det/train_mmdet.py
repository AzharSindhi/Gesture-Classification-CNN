# Import necessary modules
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv import Config
import os
# Set random seed for reproducibility
set_random_seed(0)


# Define the configuration file for the model
__base__ = "net/cluster/azhar/olfactory-gestures-azhar/Gesture Classification CNN/person_det/faster_rcnn_r50_fpn_coco.py"
cfg = Config.fromfile(
    '/net/cluster/azhar/olfactory-gestures-azhar/Gesture Classification CNN/person_det/faster_rcnn_r50_fpn_coco.py')

# Modify the configuration file to work with your dataset
cfg.dataset_type = 'CocoDataset'
cfg.data_root = '/net/cluster/azhar/datasets/peopleart_merged/'
cfg.data.train.img_prefix = cfg.data_root + "data/person"
cfg.data.train.classes = ('background', 'person')
cfg.data.train.ann_file = os.path.join(
    cfg.data_root, 'train_merged.json')

cfg.data.val.img_prefix = cfg.data_root + "data/person"
cfg.data.val.ann_file = os.path.join(
    cfg.data_root, 'test_merged.json')

cfg.data.val.classes = ('background', 'person')
cfg.data.test.img_prefix = cfg.data_root + "data/person"
cfg.data.test.ann_file = os.path.join(
    cfg.data_root, 'test_merged.json')
cfg.data.test.classes = ('background', 'person')
cfg.model.roi_head.bbox_head.num_classes = 2
# cfg.optimizer.lr = 0.02
cfg.total_epochs = 20
cfg.checkpoint_config.interval = 1
# cfg.log_config.interval = 10
# cfg.evaluation.interval = 2
cfg.gpu_ids = [0]
cfg.data.samples_per_gpu = 4
# cfg.data.samples_per_gpu = 20
# cfg.data.test.samples_per_gpu = 20
cfg.seed = 444
cfg.device = 'cuda'
cfg.work_dir = './model_results'
# Build the dataset and model
# dataset = build_dataset(cfg.data.train)
# model = build_detector(cfg.model)


# # Train the model
# train_detector(
#     model, dataset, cfg, distributed=False, validate=True)
