# ---------------------------------------------------------------
# Ablation C1: Sobel Boundary Loss Mode
# Compares axis-aligned Sobel gradient boundary detection
# against the baseline affinity mode.
# Sobel is sensitive to orientation — may underperform on
# satellite imagery where boundaries have arbitrary directions.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    boundary_loss_mode='sobel',
    # Sobel mode uses standard boundary loss, not AffinityBoundaryLoss
    affinity_loss=None,
)

name = 'ablation_C1_boundary_sobel'
exp = 'ablation_ssl'
