# ---------------------------------------------------------------
# Ablation A3: Remove Pseudo-Label Warmup
# Tests the contribution of linear warmup for pseudo-labels.
# Setting pseudo_label_warmup_iters=0 applies full pseudo-label
# weight from iteration 0.
# Expected: noisy gradients early in training, potential
#           confirmation bias from unreliable teacher predictions.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    pseudo_label_warmup_iters=0,
)

name = 'ablation_A3_no_warmup'
exp = 'ablation_ssl'
