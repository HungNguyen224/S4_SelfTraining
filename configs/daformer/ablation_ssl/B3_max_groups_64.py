# ---------------------------------------------------------------
# Ablation B3b: Max Groups = 64
# Hyperparameter sensitivity: moderate prototype groups.
# Baseline uses max_groups=96.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=64,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
    ),
)

name = 'ablation_B3_max_groups_64'
exp = 'ablation_ssl'
