# ---------------------------------------------------------------
# Ablation B1c: Boundary Lambda = 0.7
# Hyperparameter sensitivity: high boundary loss weight.
# Baseline uses boundary_lambda=0.5.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    boundary_lambda=0.7,
)

name = 'ablation_B1_boundary_lambda_07'
exp = 'ablation_ssl'
