# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# modified by xjj 0.02->0.01 1 GPU and 8 imgs per GPU
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12, 24, 40,60,100])  # step=[8, 11]
total_epochs = 48*3
