# ---------------------------------------------------------------
# Ablation C2: Laplacian Boundary Loss Mode
# Compares isotropic Laplacian edge detection against baseline
# affinity mode. Laplacian is rotation-invariant like affinity,
# but operates on second-order gradients and may be more
# sensitive to noise in satellite imagery.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    boundary_loss_mode='laplacian',
    affinity_loss=None,
)

name = 'ablation_C2_boundary_laplacian'
exp = 'ablation_ssl'
