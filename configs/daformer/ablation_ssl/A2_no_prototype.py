# ---------------------------------------------------------------
# Ablation A2: Remove Prototype Grouping Loss (DAPG)
# Tests the contribution of DynamicAnchor + DAPGLoss.
# Setting proto_lambda=0 disables prototype clustering supervision.
# Expected: reduced intra-class compactness, worse performance
#           on classes with high intra-class diversity (vegetation,
#           agriculture with varying crop types).
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    proto_lambda=0.0,
)

name = 'ablation_A2_no_prototype'
exp = 'ablation_ssl'
