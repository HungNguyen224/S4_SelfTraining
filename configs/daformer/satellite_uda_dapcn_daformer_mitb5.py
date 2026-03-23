# ---------------------------------------------------------------
# DAPCN + DAFormer for Satellite Image UDA
#
# Architecture: DAFormer (SepASPP decoder + MiT-B5 encoder)
# UDA Method:   DAPCN (boundary-aware + prototype grouping)
# Boundary:     Affinity mode (orientation-invariant, suitable for
#               satellite imagery with no repeatable shapes)
#
# Usage:
#   python tools/train.py configs/daformer/satellite_uda_dapcn_daformer_mitb5.py
#
# Before training:
#   1. Update data paths in configs/_base_/datasets/uda_satellite_512x512.py
#   2. Update num_classes in the model config override below
#   3. Place MiT-B5 pretrained weights at pretrained/mit_b5.pth
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture (MiT-B5 encoder + SepASPP decoder)
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # Satellite UDA Data Loading
    '../_base_/datasets/uda_satellite_512x512.py',
    # DAPCN UDA Self-Training
    '../_base_/uda/dapcn.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Warmup + Polynomial Decay
    '../_base_/schedules/poly10warm.py'
]

# Random Seed
seed = 0

# --- Model: update num_classes for your satellite dataset ---
model = dict(
    decode_head=dict(
        num_classes=7,  # Match your dataset's num_classes
    ))

# --- DAPCN UDA Configuration ---
uda = dict(
    # EMA teacher momentum
    alpha=0.999,
    # Pseudo-label confidence threshold
    pseudo_threshold=0.968,
    # No top/bottom ignore for satellite (overhead view, no rectification)
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    # --- DAPCN loss weights ---
    # Boundary loss: higher weight for satellite (many co-occurring classes
    # with irregular boundaries)
    boundary_lambda=0.5,
    # Prototype grouping loss
    proto_lambda=0.1,
    # --- Boundary mode ---
    # 'affinity' for satellite (orientation-invariant, pairwise relational)
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_proto_on_target=True,
    # --- DynamicAnchor: larger max_groups for satellite diversity ---
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=96,      # More groups for satellite intra-class diversity
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
    ),
    # --- DAPGLoss: tighter margin for satellite ---
    dapg_loss=dict(
        type='DAPGLoss',
        margin=0.3,
        lambda_inter=0.5,
        lambda_quality=0.1,
        loss_weight=1.0,
    ),
    # --- AffinityBoundaryLoss ---
    affinity_loss=dict(
        type='AffinityBoundaryLoss',
        temperature=0.5,
        scale=2,
        num_neighbors=4,
        ignore_index=255,
        loss_weight=1.0,
    ),
)

# --- Rare Class Sampling (critical for satellite class imbalance) ---
data = dict(
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))

# --- Optimizer ---
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

# --- Logging & Checkpointing ---
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

# --- Meta Information ---
name = 'satellite_uda_dapcn_daformer_mitb5'
exp = 'satellite'
name_dataset = 'satellite_uda'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_affinity_rcs'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
