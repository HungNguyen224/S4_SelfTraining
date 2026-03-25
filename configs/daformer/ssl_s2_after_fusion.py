# ---------------------------------------------------------------
# Solution 2: DynAnchor After SepASPP Fusion (Decoder Space)
#
# DynamicAnchorModule operates on the fused decoder features (256-d)
# produced by DAFormerHead._fuse_features(). Prototypes live natively
# in the same space as conv_seg — no projection layer needed.
#
# Pros:
#   - Prototypes capture multi-scale contextual information from the
#     full SepASPP fusion pipeline (embed 4 scales → concat → fuse)
#   - No extra learnable projection layer — fewer parameters
#   - Prototype → conv_seg path is geometrically consistent:
#     both operate in the same 256-d manifold
#   - Correction formula is self-consistent: assignment, prototypes,
#     and classifier all in the same space
# Cons:
#   - Slightly more computation per iteration (fuse_features called
#     twice for unlabeled images: once in teacher, once for DynAnchor)
#
# Usage:
#   python tools/train.py configs/daformer/ssl_s2_after_fusion.py
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/ssl_satellite_512x512.py',
    '../_base_/ssl/dapcn_ssl.py',
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
    # anchor_after_fusion=True → DynAnchor on fused decoder features (256-d)
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
    boundary_lambda=0.5,
    proto_lambda=0.1,
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_proto_on_target=True,

    # DynamicAnchor — feature_dim auto-detected as 256 (decoder fused space)
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

checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'ssl_s2_after_fusion'
exp = 'satellite_ssl'
name_dataset = 'satellite_ssl'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_ssl_s2_after_fusion'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
