# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    # pretrained=None,
    roi_head=dict(
        bbox_head=dict(num_classes=2)
        ))

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('kuiyang', 'xuezhong')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix='data/coco_2021/train2021/',
        classes=classes,
        ann_file='data/coco_2021/annotations/instances_train.json'),
    val=dict(
        type=dataset_type,
        img_prefix='data/coco_2021/val2021/',
        classes=classes,
        ann_file='data/coco_2021/annotations/instances_val.json'),
    test=dict(
        type=dataset_type,
        img_prefix='data/coco_2021/train2021/',
        classes=classes,
        ann_file='data/coco_2021/annotations/instances_train.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
