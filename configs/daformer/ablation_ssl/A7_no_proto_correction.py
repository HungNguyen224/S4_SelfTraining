# ---------------------------------------------------------------
# Ablation A7: Remove Prototype-Based Pseudo-Label Correction
# Tests the contribution of the prototype correction step:
#   p^c_j = sum_i f_theta(PT_i) * a_ij
# Setting proto_correction=False disables the correction entirely.
# Pseudo-labels rely solely on the EMA teacher's predictions.
# Expected: noisier pseudo-labels, particularly on pixels whose
#           features are close to well-learned prototypes but where
#           the teacher prediction is incorrect.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    proto_correction=False,
)

name = 'ablation_A7_no_proto_correction'
exp = 'ablation_ssl'
