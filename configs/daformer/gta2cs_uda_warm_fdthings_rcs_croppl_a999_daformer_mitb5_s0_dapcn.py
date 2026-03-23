# ---------------------------------------------------------------
# DEPRECATED: This config is kept for reference only.
# Use satellite_uda_dapcn_daformer_mitb5.py for new experiments.
# ---------------------------------------------------------------
# Original: DAFormer + DAPCN for GTA -> Cityscapes UDA
# This config retains ImageNet Feature Distance which has been
# removed from the refactored DAPCN for satellite use.
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    '../_base_/uda/dapcn.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0

uda = dict(
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    boundary_lambda=0.5,
    proto_lambda=0.1,
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_proto_on_target=True,
)

data = dict(
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))

optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)

checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

name = 'gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_dapcn'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_a999_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
