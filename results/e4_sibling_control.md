# E4 per-prompt centring, clusters of size < 3 gated to the uncentred probe (qwen05; dev set, prompt-grouped 5-fold OOF, n=2160)

| config                                                |   dev_BA | folds                               | boot95         | fold_delta_vs_E3_blend_pp           |
|:------------------------------------------------------|---------:|:------------------------------------|:---------------|:------------------------------------|
| E3 blend (uncentred hidden), reference                |   0.4125 | [0.41, 0.431, 0.396, 0.426, 0.4]    | [0.392, 0.433] | [0.0, 0.0, 0.0, 0.0, 0.0]           |
| hidden only, uncentred (E3)                           |   0.3657 | [0.387, 0.354, 0.343, 0.359, 0.387] | [0.344, 0.387] | [-2.31, -7.64, -5.32, -6.71, -1.39] |
| hidden only, oracle-centred                           |   0.4935 | [0.514, 0.505, 0.458, 0.495, 0.495] | [0.473, 0.516] | [10.42, 7.41, 6.25, 6.94, 9.49]     |
| hidden only, cluster-centred (gated)                  |   0.488  | [0.512, 0.505, 0.451, 0.479, 0.493] | [0.468, 0.511] | [10.19, 7.41, 5.56, 5.32, 9.26]     |
| blend: oracle-centred hidden (nested weights)         |   0.5083 | [0.523, 0.528, 0.481, 0.507, 0.502] | [0.489, 0.528] | [11.34, 9.72, 8.56, 8.1, 10.19]     |
| blend: cluster-centred hidden, gated (nested weights) |   0.5046 | [0.512, 0.528, 0.484, 0.502, 0.498] | [0.484, 0.525] | [10.19, 9.72, 8.8, 7.64, 9.72]      |

Clustering (word 1-2gram TF-IDF, average-link cosine): threshold per fold [0.95, 0.95, 0.95, 0.95, 0.95], val-fold ARI vs prompt_id [0.988, 1.0, 0.982, 0.971, 0.991], gated rows across folds 11/2160; pooled 2160-row ARI 0.962. tau per fold [1.0, 0.0, 0.0, 0.0, 1.0].

## Sibling-count control (k passages per prompt kept in the validation fold, 5 seeds)
|   passages_per_prompt |   BA_uncentred_E3_blend |   BA_cluster_centred_gated_blend |   delta_pp |   frac_gated_rows |
|----------------------:|------------------------:|---------------------------------:|-----------:|------------------:|
|                     1 |                  0.418  |                           0.4257 |       0.78 |             0.793 |
|                     2 |                  0.4337 |                           0.4357 |       0.2  |             0.861 |
|                     3 |                  0.4162 |                           0.4445 |       2.83 |             0.029 |
|                     6 |                  0.4256 |                           0.4972 |       7.16 |             0.014 |
|                    18 |                  0.4125 |                           0.5046 |       9.21 |             0.005 |

## Decision
gated cluster-centred blend vs E3 blend: **+9.21 pp**; per-fold [10.19, 9.72, 8.8, 7.64, 9.72]; k=1 delta +0.78 pp, k=2 delta +0.20 pp (guard ok); oracle-centred blend +9.58 pp. The gated variant passed its revised rule.
