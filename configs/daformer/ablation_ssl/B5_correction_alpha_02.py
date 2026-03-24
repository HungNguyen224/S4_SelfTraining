# ---------------------------------------------------------------
# Ablation B5a: Proto Correction Alpha = 0.2
# Hyperparameter sensitivity: low prototype blending weight.
# Baseline uses proto_correction_alpha=0.5.
# Lower alpha → teacher-dominated, less prototype influence.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    proto_correction_alpha=0.2,
)

name = 'ablation_B5_correction_alpha_02'
exp = 'ablation_ssl'
