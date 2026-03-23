# ---------------------------------------------------------------
# Ablation C3: Hybrid Boundary Loss Mode
# Combines binary boundary map loss with AffinityBoundaryLoss.
# hybrid_binary_weight controls the weighting between:
#   (1 - w) * affinity_loss + w * binary_boundary_loss
# Tests whether explicit boundary supervision complements
# the pairwise affinity approach.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    boundary_loss_mode='hybrid',
    hybrid_binary_weight=0.5,
)

name = 'ablation_C3_boundary_hybrid'
exp = 'ablation_ssl'
