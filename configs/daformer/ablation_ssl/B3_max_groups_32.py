# ---------------------------------------------------------------
# Ablation B3a: Max Groups = 32
# Hyperparameter sensitivity: fewer prototype groups.
# Baseline uses max_groups=96.
# Fewer groups may under-represent intra-class diversity in
# satellite land cover (e.g., different vegetation types).
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=32,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
    ),
)

name = 'ablation_B3_max_groups_32'
exp = 'ablation_ssl'
