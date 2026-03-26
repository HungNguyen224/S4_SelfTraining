# ---------------------------------------------------------------
# Solution 2: DynAnchor After DeepLabV3+ Fusion (Decoder Space)
# Backbone: ResNet-101 (d8)  |  Decoder: DeepLabV3+
#
# DynamicAnchorModule operates on the fused decoder features (512-d)
# produced by DepthwiseSeparableASPPHead._fuse_features().
# Prototypes live natively in the same space as conv_seg — no
# projection layer needed.
#
# Dimension trace:
#   Encoder: ResNet-101 stages [256, 512, 1024, 2048]
#   ASPP:    2048 → 5×512 → cat 2560 → bottleneck 512
#   c1:      256 → 48
#   Fusion:  cat(512, 48) = 560 → sep_bottleneck → 512
#   DynAnchor input: 512-d (fused decoder features)
#   conv_seg: Conv2d(512, num_classes, 1) — directly compatible
#
# Pros:
#   - Prototypes capture ASPP multi-scale context + low-level detail
#   - No extra learnable projection — fewer parameters
#   - Prototype → conv_seg geometrically consistent (same 512-d space)
# Cons:
#   - DAPG loss gradients flow through ASPP + c1 + sep_bottleneck,
#     adding gradient load to decoder layers (compensated by halving
#     proto_lambda)
#
# Usage:
#   python tools/train.py configs/daformer/ssl_s2_deeplabv3plus_r101.py
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/deeplabv3plus_r101-d8.py',
    '../_base_/datasets/ssl_satellite_512x512.py',
    '../_base_/ssl/dapcn_ssl_dlv3plus.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0

# --- Model ---
model = dict(
    decode_head=dict(
        num_classes=7,
    ))

# --- DAPCN-SSL: Solution 2 (anchor_after_fusion=True) ---
uda = dict(
    # anchor_after_fusion=True → DynAnchor on fused decoder features (512-d)
    # Prototypes are natively in conv_seg's input space, no projection
    anchor_after_fusion=True,

    # EMA teacher
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,

    # SSL warmup
    pseudo_label_warmup_iters=1000,

    # Prototype-based pseudo-label correction
    proto_correction=True,
    proto_correction_alpha=0.5,
    proto_correction_start_iter=1000,

    # DAPCN loss weights
    # NOTE: In Solution 2, DAPG loss gradients flow through
    # _fuse_features() (ASPP + c1 + sep_bottleneck), adding a second
    # gradient source to the decoder layers on top of the segmentation
    # loss.  Lower proto_lambda to compensate for the ~2-3x higher
    # effective gradient on those shared layers compared to Solution 1
    # (where DAPG bypasses the decoder entirely).
    boundary_lambda=0.5,
    proto_lambda=0.05,
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_proto_on_target=True,

    # DynamicAnchor — feature_dim auto-detected as 512 (decoder fused space)
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=96,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
    ),
    # DAPGLoss
    dapg_loss=dict(
        type='DAPGLoss',
        margin=0.3,
        lambda_inter=0.5,
        lambda_quality=0.1,
        loss_weight=1.0,
    ),
    # AffinityBoundaryLoss
    affinity_loss=dict(
        type='AffinityBoundaryLoss',
        temperature=0.5,
        scale=2,
        num_neighbors=4,
        ignore_index=255,
        loss_weight=1.0,
    ),
)

# --- Rare Class Sampling ---
data = dict(
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))

# --- Optimizer ---
# DynAnchor prototypes: no weight decay (cluster centres), 10x LR.
# quality_net: 10x LR to match the decode head.
# Note: Solution 2 has no proto_to_decoder (prototypes already 512-d).
#
# GRADIENT COUPLING WARNING (Solution 2 only):
# In S2, the decode head layers (ASPP, c1_bottleneck, sep_bottleneck)
# receive gradients from BOTH the segmentation loss AND the DAPG loss
# (via _fuse_features).  In S1, DAPG bypasses the decoder entirely.
# We compensate by halving proto_lambda (0.05 vs 0.1) rather than
# changing lr_mult.
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'dynamic_anchor.prototypes': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.quality_net': dict(lr_mult=10.0),
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)

checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'ssl_s2_deeplabv3plus_r101'
exp = 'satellite_ssl'
name_dataset = 'satellite_ssl'
name_architecture = 'deeplabv3plus_r101_d8'
name_encoder = 'r101'
name_decoder = 'deeplabv3plus'
name_uda = 'dapcn_ssl_s2_after_fusion'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
