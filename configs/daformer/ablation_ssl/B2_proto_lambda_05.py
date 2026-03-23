# ---------------------------------------------------------------
# Ablation B2c: Proto Lambda = 0.5
# Hyperparameter sensitivity: high prototype loss weight.
# Baseline uses proto_lambda=0.1.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    proto_lambda=0.5,
)

name = 'ablation_B2_proto_lambda_05'
exp = 'ablation_ssl'
