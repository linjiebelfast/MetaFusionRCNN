# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2),
        mask_head=dict(num_classes=2)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('kuiyang', 'xuezhong')
data = dict(
    samples_per_gpu=8,
    train=dict(
        img_prefix='data/coco/train2020/',
        classes=classes,
        ann_file='data/coco/annotations/instances_train.json'),
    val=dict(
        img_prefix='data/coco/val2020/',
        classes=classes,
        ann_file='data/coco/annotations/instances_val.json'),
    test=dict(
        img_prefix='data/coco/val2020/',
        classes=classes,
        ann_file='data/coco/annotations/instances_val.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
