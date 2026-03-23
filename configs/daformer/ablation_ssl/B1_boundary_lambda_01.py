# ---------------------------------------------------------------
# Ablation B1a: Boundary Lambda = 0.1
# Hyperparameter sensitivity: low boundary loss weight.
# Baseline uses boundary_lambda=0.5.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    boundary_lambda=0.1,
)

name = 'ablation_B1_boundary_lambda_01'
exp = 'ablation_ssl'
