_base_ = [
    './_base_/default_runtime.py',
    './_base_/models/daformer_sepaspp_mitb5.py',
]

dataset_type = 'CityscapesDataset'
data_root = 'data/reallabels/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        pipeline=test_pipeline))

model = dict(pretrained=None)

exp = 'test_reallabels'
name = 'reallabels_test'
seed = 0
gpu_ids = range(1)
total_iters = 100
optimizer = dict(type='AdamW', lr=6e-5)
optimizer_config = None
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(by_epoch=False, interval=10)
workflow = [('train', 10)]
evaluation = dict(interval=5, metric='mIoU')
