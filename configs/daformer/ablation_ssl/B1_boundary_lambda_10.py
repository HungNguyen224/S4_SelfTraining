# ---------------------------------------------------------------
# Ablation B1d: Boundary Lambda = 1.0
# Hyperparameter sensitivity: very high boundary loss weight.
# Baseline uses boundary_lambda=0.5.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    boundary_lambda=1.0,
)

name = 'ablation_B1_boundary_lambda_10'
exp = 'ablation_ssl'
