# MARCEL benchmark: kraken

- Models trained: 5
- Seeds: 1
- Epochs (max): 10, patience: 5
- Mode (two-state): `parallel`
- Equalization mode: `none`
- Hidden (default): 128, Layers: 3

## Test MAE (lower is better)

| model | hidden | params | per-target MAE | overall MAE |
|---|---:|---:|---|---|
| GraphSetConv-PaiNN-Broadcast | 128 | 1,382,789 | sterimol_B5: 0.420±nan | 0.4198 ± nan |
| GraphSetConv-PaiNN-CrossAttn | 128 | 1,581,701 | sterimol_B5: 0.472±nan | 0.4716 ± nan |
| PaiNN+DeepSets | 128 | 802,817 | sterimol_B5: 0.448±nan | 0.4483 ± nan |
| PaiNN+SetTransformer | 128 | 935,681 | sterimol_B5: 0.408±nan | **0.4083 ± nan** |
| PaiNN+BoltzmannPool | 128 | 769,793 | sterimol_B5: 0.429±nan | 0.4291 ± nan |

## Pairwise Wilcoxon signed-rank tests (Holm-Bonferroni adjusted)

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|
