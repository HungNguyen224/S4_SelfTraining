# ---------------------------------------------------------------
# Ablation A1: Remove Boundary Loss
# Tests the contribution of AffinityBoundaryLoss.
# Setting boundary_lambda=0 disables boundary-aware supervision.
# Expected: degraded boundary delineation, lower mIoU on
#           thin/elongated classes (road, water).
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    boundary_lambda=0.0,
)

name = 'ablation_A1_no_boundary'
exp = 'ablation_ssl'
