_base_ = [
    '_base_/default_runtime.py',
    '_base_/models/daformer_sepaspp_mitb5.py',
    '_base_/datasets/uda_oem_512x512.py',
    '_base_/uda/dapcn.py',
    '_base_/schedules/adamw.py',
    '_base_/schedules/poly10warm.py',
]

model = dict(
    pretrained=None,
    decode_head=dict(num_classes=9),  # OEM has 9 classes: 0-8
)

optimizer = dict(type='AdamW', lr=6e-5)
optimizer_config = None
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=2000)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])

evaluation = dict(interval=200, metric='mIoU')

checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=2)

workflow = [('train', 2000)]

exp = 'oem_test'
name = 'oem_dapcn_test'
seed = 0
gpu_ids = range(1)
