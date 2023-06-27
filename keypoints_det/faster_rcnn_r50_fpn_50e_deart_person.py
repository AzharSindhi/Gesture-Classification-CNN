_base_ = [
    '/net/cluster/azhar/mmdetection-ODOR/myconfs/odor3/faster_rcnn_r50_fpn.py',
    '/net/cluster/azhar/mmdetection-ODOR/configs/_base_/datasets/deart_instance_person.py',
    '/net/cluster/azhar/mmdetection-ODOR/myconfs/odor3/schedule_50e.py', '/net/cluster/azhar/mmdetection-ODOR/configs/_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        )
    )
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4
)
