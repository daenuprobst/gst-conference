# USPTO yield benchmark

- Models: 4
- Seeds: 1
- Epochs (max): 500, patience: 20
- Equalization mode: `smaller`
- Feature mode: `rich`
- Max reactions (subsample): 6000
- Hidden (default): 128, Layers: 3

## Test metrics (mean ± 95% CI)

| model | hidden | params | RMSE (%) | MAE (%) | R² |
|---|---:|---:|---|---|---|
| GraphSetConv-Broadcast | 56 | 131,212 | 26.883 ± nan | 21.696 ± nan | -0.044 ± nan |
| GraphSetConv-CrossAttn | 52 | 146,904 | 26.390 ± nan | 21.298 ± nan | **-0.006 ± nan** |
| GCN+DeepSets | 128 | 137,857 | 26.525 ± nan | 21.363 ± nan | -0.017 ± nan |
| GCN+SetTransformer | 128 | 270,721 | 26.867 ± nan | 21.975 ± nan | -0.043 ± nan |

## Pairwise Wilcoxon on RMSE (Holm-Bonferroni adjusted)

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|
