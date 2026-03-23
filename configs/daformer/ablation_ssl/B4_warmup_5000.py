# ---------------------------------------------------------------
# Ablation B4c: Pseudo-Label Warmup = 5000 iters
# Hyperparameter sensitivity: very long warmup (12.5% of training).
# Baseline uses pseudo_label_warmup_iters=1000.
# May be beneficial when labeled ratio is very small (<5%).
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    pseudo_label_warmup_iters=5000,
)

name = 'ablation_B4_warmup_5000'
exp = 'ablation_ssl'
