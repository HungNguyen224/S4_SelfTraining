# ---------------------------------------------------------------
# Ablation B2b: Proto Lambda = 0.2
# Hyperparameter sensitivity: moderate prototype loss weight.
# Baseline uses proto_lambda=0.1.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    proto_lambda=0.2,
)

name = 'ablation_B2_proto_lambda_02'
exp = 'ablation_ssl'
