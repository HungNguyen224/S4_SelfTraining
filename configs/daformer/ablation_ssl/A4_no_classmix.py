# ---------------------------------------------------------------
# Ablation A4: Remove ClassMix Augmentation
# Tests the contribution of ClassMix data mixing.
# Setting mix='none' disables class-based cut-paste mixing.
# Expected: reduced regularization, worse pseudo-label quality,
#           the model may overfit to labeled data distribution.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    mix='none',
)

name = 'ablation_A4_no_classmix'
exp = 'ablation_ssl'
