# ---------------------------------------------------------------
# Ablation B4a: Pseudo-Label Warmup = 500 iters
# Hyperparameter sensitivity: shorter warmup period.
# Baseline uses pseudo_label_warmup_iters=1000.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    pseudo_label_warmup_iters=500,
)

name = 'ablation_B4_warmup_500'
exp = 'ablation_ssl'
