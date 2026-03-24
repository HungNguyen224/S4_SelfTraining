# ---------------------------------------------------------------
# Ablation B5b: Proto Correction Alpha = 0.8
# Hyperparameter sensitivity: high prototype blending weight.
# Baseline uses proto_correction_alpha=0.5.
# Higher alpha → prototype-dominated, less teacher influence.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    proto_correction_alpha=0.8,
)

name = 'ablation_B5_correction_alpha_08'
exp = 'ablation_ssl'
