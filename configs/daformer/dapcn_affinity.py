# ---------------------------------------------------------------
# DAFormer with DAPCN (Boundary-Aware Prototype Contrastive Network)
# Using Affinity Boundary Loss
# GTA -> Cityscapes UDA
# Reference: docs/DAPCN.md
# --------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # DAPCN UDA Self-Training (replaces DACS)
    '../_base_/uda/dapcn.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]

# Random Seed
seed = 0

# Modifications to DAPCN UDA
uda = dict(
    # Increased Alpha (EMA momentum)
    alpha=0.999,
    # Pseudo-Label Threshold
    pseudo_threshold=0.968,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    # Thing-Class Feature Distance (optional, set to 0 to disable)
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    # DAPCN-specific parameters
    boundary_lambda=0.5,
    contrastive_lambda=0.1,
    contrastive_temp=0.07,
    prototype_ema=0.999,
    proto_feature_dim=256,
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_contrastive_on_target=True,
    apply_prototype_on_target=True,
    # Affinity Boundary Loss (MMSegmentation style)
    boundary_loss_mode='affinity',
    affinity_loss=dict(
        type='AffinityBoundaryLoss',
        temperature=0.5,
        scale=2,
        num_neighbors=4,
        ignore_index=255,
        loss_weight=1.0,
    ),
)

data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))

# Optimizer Hyperparameters
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

# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

# Meta Information for Result Analysis
name = 'gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_dapcn_affinity'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_a999_fd_things_rcs0.01_affinity'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
