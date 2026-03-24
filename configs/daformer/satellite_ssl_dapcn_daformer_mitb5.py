# ---------------------------------------------------------------
# DAPCN-SSL + DAFormer for Satellite Semi-Supervised Segmentation
#
# Architecture: DAFormer (SepASPP decoder + MiT-B5 encoder)
# SSL Method:   DAPCN-SSL (boundary-aware + prototype grouping)
# Boundary:     Affinity mode (orientation-invariant)
#
# Usage:
#   python tools/train.py configs/daformer/satellite_ssl_dapcn_daformer_mitb5.py
#
# Before training:
#   1. Update data_root in configs/_base_/datasets/ssl_satellite_512x512.py
#   2. Create splits/labeled.txt with labeled image basenames
#   3. Update num_classes below to match your taxonomy
#   4. Place MiT-B5 pretrained weights at pretrained/mit_b5.pth
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # Semi-Supervised Satellite Data Loading
    '../_base_/datasets/ssl_satellite_512x512.py',
    # DAPCN-SSL Self-Training
    '../_base_/ssl/dapcn_ssl.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Warmup + Polynomial Decay
    '../_base_/schedules/poly10warm.py'
]

# Random Seed
seed = 0

# --- Model: update num_classes for your dataset ---
model = dict(
    decode_head=dict(
        num_classes=7,
    ))

# --- DAPCN-SSL Configuration ---
uda = dict(
    # EMA teacher momentum (higher = more stable teacher)
    alpha=0.999,
    # Pseudo-label confidence threshold
    pseudo_threshold=0.968,
    # No top/bottom ignore (satellite = overhead view)
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    # --- SSL-specific: pseudo-label warmup ---
    # Ramp up pseudo-label weight over 1000 iters; critical when
    # only a small fraction of images are labeled — the teacher is
    # unreliable early on and needs time to learn from labeled data
    pseudo_label_warmup_iters=1000,
    # --- Prototype-based pseudo-label correction ---
    # p^c_j = sum_i f_theta(PT_i) * a_ij
    # Prototypes provide class-agnostic structural knowledge that
    # corrects noisy teacher predictions via feature similarity
    proto_correction=True,
    proto_correction_alpha=0.5,
    proto_correction_start_iter=1000,
    # --- DAPCN loss weights ---
    boundary_lambda=0.5,
    proto_lambda=0.1,
    # --- Boundary: affinity (orientation-invariant) ---
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_proto_on_target=True,
    # --- DynamicAnchor ---
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=96,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
    ),
    # --- DAPGLoss ---
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

# --- Rare Class Sampling (on labeled subset) ---
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
name = 'satellite_ssl_dapcn_daformer_mitb5'
exp = 'satellite_ssl'
name_dataset = 'satellite_ssl'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_ssl_affinity_rcs'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
