# ---------------------------------------------------------------
# Ablation B4b: Pseudo-Label Warmup = 2000 iters
# Hyperparameter sensitivity: longer warmup period.
# Baseline uses pseudo_label_warmup_iters=1000.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    pseudo_label_warmup_iters=2000,
)

name = 'ablation_B4_warmup_2000'
exp = 'ablation_ssl'
