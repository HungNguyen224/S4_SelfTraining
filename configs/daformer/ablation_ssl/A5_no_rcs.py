# ---------------------------------------------------------------
# Ablation A5: Remove Rare Class Sampling (RCS)
# Tests the contribution of temperature-based class-aware sampling.
# Removing RCS reverts to uniform random sampling on labeled set.
# Expected: severe performance drop on minority classes (barren,
#           water) while majority classes remain stable.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

# Override data without rare_class_sampling
data = dict(
    train=dict(
        rare_class_sampling=None))

name = 'ablation_A5_no_rcs'
exp = 'ablation_ssl'
