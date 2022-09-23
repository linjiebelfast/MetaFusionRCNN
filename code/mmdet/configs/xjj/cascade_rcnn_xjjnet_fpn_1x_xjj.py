# model settings
model = dict(
    type='CascadeRCNN',
    # pretrained='torchvision://resnet50',
    backbone=dict(
        # type='OriginResNet',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'
        ),
    neck=dict(
        type='FPN',
        in_channels=[256+4, 512+4, 1024+4, 2048+4],
        out_channels=256+0,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256+0,
        feat_channels=256+0,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256+0,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256+0,
                fc_out_channels=1024,
                roi_feat_size=7,
                # num_classes=80,
                num_classes=2,  # modified by xujj
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256+0,
                fc_out_channels=1024,
                roi_feat_size=7,
                # num_classes=80,
                num_classes=2,  # modified by xujj
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256+0,
                fc_out_channels=1024,
                roi_feat_size=7,
                # num_classes=80,
                num_classes=2,  # modified by xujj
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        # nms_thr=0.7,
        nms_thr=0.3,
        min_bbox_size=0),
    # rcnn=[
    #     dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.5,
    #             neg_iou_thr=0.5,
    #             min_pos_iou=0.5,
    #             match_low_quality=False,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=512,
    #             pos_fraction=0.25,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=True),
    #         pos_weight=-1,
    #         debug=False),
    #     dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.6,
    #             neg_iou_thr=0.6,
    #             min_pos_iou=0.6,
    #             match_low_quality=False,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=512,
    #             pos_fraction=0.25,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=True),
    #         pos_weight=-1,
    #         debug=False),
    #     dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.7,
    #             neg_iou_thr=0.7,
    #             min_pos_iou=0.7,
    #             match_low_quality=False,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=512,
    #             pos_fraction=0.25,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=True),
    #         pos_weight=-1,
    #         debug=False)
    # ]
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.4,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weight=[1, 0.5, 0.25]
    )
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        # nms_thr=0.7,
        nms_thr=0.3,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        # nms=dict(type='nms', iou_threshold=0.5),
        nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001),
        max_per_img=100))


############################
#  dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(  # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    mean=[66.026, 44.032, 49.044], std=[81.543, 57.91, 56.214], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(568, 760), (298, 400)], keep_ratio=True),  # modified by xujj:img_scale=(1333, 800)
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(568, 760),  # modified by xujj:img_scale=(1333, 800)
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ('kuiyang', 'xuezhong')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        img_prefix='data/coco_2021/train2021/',
        classes=classes,
        ann_file='data/coco_2021/annotations/instances_train.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix='data/coco_2021/val2021/',
        classes=classes,
        ann_file='data/coco_2021/annotations/instances_val.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # img_prefix='data/coco_2021/val2021/',
        img_prefix='data/coco_2021/train2021/',
        classes=classes,
        # ann_file='data/coco_2021/annotations/instances_val.json',
        ann_file='data/coco_2021/annotations/instances_train.json',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

####################################
# optimizer settings
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.0025, weight_decay=0.0)
# modified by xjj 0.02->0.01 1 GPU and 8 imgs per GPU
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[8, 16, 25, 31])  # step=[8, 11]
total_epochs = 36

#########################
# runtime settings
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
# load_from = 'meta_fusionDir/epoch_36.pth'
# load_from = None
# resume_from = 'baseDir/epoch_36.pth'
resume_from = None
workflow = [('train', 1)]

