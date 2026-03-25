# ---------------------------------------------------------------
# Solution 1: DynAnchor in Encoder Space + Linear Projection
#
# DynamicAnchorModule operates on raw encoder features (512-d).
# A learned Linear(512→256) projects prototypes to the decoder's
# fused space before passing through conv_seg for pseudo-label
# correction.
#
# Pros:
#   - Prototypes capture single-scale encoder representations
#   - DAPGLoss shapes prototypes in the same space as encoder output
# Cons:
#   - Extra learnable layer (512→256) that must co-train
#   - Prototypes do not benefit from multi-scale context fusion
#
# Usage:
#   python tools/train.py configs/daformer/ssl_s1_projection.py
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

# --- DAPCN-SSL: Solution 1 (anchor_after_fusion=False) ---
uda = dict(
    # anchor_after_fusion=False → DynAnchor on encoder features (512-d)
    # proto_to_decoder = Linear(512, 256) bridges to conv_seg
    anchor_after_fusion=False,

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

    # DynamicAnchor — feature_dim auto-detected as 512 (encoder space)
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
# DynAnchor prototypes: no weight decay (cluster centres should not be
# pulled toward zero) and 10x LR (randomly initialised, and gradients
# are attenuated through 3 EM iterations).
# proto_to_decoder: same treatment (Solution 1 linear projection).
# quality_net: 10x LR to match the decode head.
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'dynamic_anchor.prototypes': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.quality_net': dict(lr_mult=10.0),
            'proto_to_decoder': dict(lr_mult=10.0, decay_mult=0.0),
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)

checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'ssl_s1_projection'
exp = 'satellite_ssl'
name_dataset = 'satellite_ssl'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_ssl_s1_projection'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
