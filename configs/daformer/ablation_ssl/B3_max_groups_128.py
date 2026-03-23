# ---------------------------------------------------------------
# Ablation B3c: Max Groups = 128
# Hyperparameter sensitivity: more prototype groups.
# Baseline uses max_groups=96.
# More groups may better capture fine-grained sub-categories
# but risk over-fragmentation with limited labeled data.
# ---------------------------------------------------------------

_base_ = ['../satellite_ssl_dapcn_daformer_mitb5.py']

uda = dict(
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=128,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
    ),
)

name = 'ablation_B3_max_groups_128'
exp = 'ablation_ssl'
