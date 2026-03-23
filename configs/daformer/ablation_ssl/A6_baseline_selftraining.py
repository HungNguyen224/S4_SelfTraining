# ---------------------------------------------------------------
# Ablation A6: Baseline Self-Training Only
# Removes ALL DAPCN components (boundary + prototype + RCS).
# Retains: EMA teacher, pseudo-labels, ClassMix, warmup.
# This is the minimal self-training baseline to measure the
# total contribution of the DAPCN framework.
# Expected: significant mIoU drop vs full model, establishing
#           the value of boundary-aware + prototype supervision.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    boundary_lambda=0.0,
    proto_lambda=0.0,
)

data = dict(
    train=dict(
        rare_class_sampling=None))

name = 'ablation_A6_baseline_selftraining'
exp = 'ablation_ssl'
