# ---------------------------------------------------------------
# Ablation B2a: Proto Lambda = 0.05
# Hyperparameter sensitivity: low prototype loss weight.
# Baseline uses proto_lambda=0.1.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    proto_lambda=0.05,
)

name = 'ablation_B2_proto_lambda_005'
exp = 'ablation_ssl'
