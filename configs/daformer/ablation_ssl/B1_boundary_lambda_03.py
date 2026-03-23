# ---------------------------------------------------------------
# Ablation B1b: Boundary Lambda = 0.3
# Hyperparameter sensitivity: moderate boundary loss weight.
# Baseline uses boundary_lambda=0.5.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    boundary_lambda=0.3,
)

name = 'ablation_B1_boundary_lambda_03'
exp = 'ablation_ssl'
