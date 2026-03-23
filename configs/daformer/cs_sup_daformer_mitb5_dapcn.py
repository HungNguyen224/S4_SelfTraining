# ---------------------------------------------------------------
# Supervised Segmentation with DAPCN Auxiliary Losses
# Dataset: Cityscapes (single-domain, fully labeled)
# Model:   DAFormer (MiT-B5 encoder + SepASPP decoder)
# Head:    DAFormerDAPCNHead (CE + boundary + DAPG + contrastive)
# ---------------------------------------------------------------
# Usage:
#   python tools/train.py configs/daformer/cs_sup_daformer_mitb5_dapcn.py
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # NOTE: Uses daformer_conv1_mitb5 as model base; we override the
    # decode_head type and fusion_cfg below to get SepASPP + DAPCN.
    '../_base_/models/daformer_conv1_mitb5.py',
    '../_base_/datasets/cityscapes_half_512x512.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]
# No UDA config imported — this is purely supervised.

seed = 0

# -- Model: override decode_head to DAFormerDAPCNHead ----------------------
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='DAFormerDAPCNHead',
        # Inherited from daformer_conv1_mitb5 base:
        #   in_channels=[64, 128, 320, 512], in_index=[0,1,2,3],
        #   channels=256, num_classes=19, dropout_ratio=0.1, etc.
        # Override fusion to SepASPP (same as daformer_sepaspp_mitb5):
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg)),
        # ------ DAPCN loss configuration ------
        # Boundary-aware loss
        boundary_lambda=0.3,
        boundary_mode='sobel',
        boundary_loss_mode='binary',
        # Dynamic anchor prototype grouping loss (persistent learnable)
        proto_lambda=0.1,
        dynamic_anchor=dict(
            type='DynamicAnchorModule',
            max_groups=64,
            temperature=0.1,
            num_iters=3,
            init_method='xavier',      # one-time init; learned via backprop
            min_quality=0.1,
            use_quality_gate=True,
            use_mask_predictor=False,
            ema_decay=0.0,             # set >0 (e.g. 0.999) for extra stability
        ),
        dapg_loss=dict(
            type='DAPGLoss',
            margin=0.3,
            lambda_inter=0.5,
            lambda_quality=0.1,
        ),
        # Memory-bank contrastive loss
        contrastive_lambda=0.1,
        contrastive_temperature=0.07,
        contrastive_sample_ratio=0.1,
        warmup_iters=500,
        num_prototypes_per_class=1,
        prototype_ema=0.999,
    ))

# -- Optimizer: 10x LR multiplier for decode head (standard practice) ------
optimizer_config = dict()
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

# -- Training schedule -----------------------------------------------------
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000)
evaluation = dict(interval=4000, metric='mIoU')
