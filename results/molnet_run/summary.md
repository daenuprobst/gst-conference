# MoleculeNet (esol) — BRICS fragment-set benchmark

- Dataset: `esol`  (regression)
- Split: `random`
- Filter K=1: `True`
- Models: 4
- Seeds: 1
- Epochs (max): 200, patience: 30
- Equalization mode: `none`
- Feature mode: `rich`
- Hidden (default): 128, Layers: 3

## Test metrics (mean ± 95% CI)

| model | hidden | params | RMSE | MAE | R² |
|---|---:|---:|---|---|---|
| GraphSetConv-Broadcast | 128 | 701,830 | 1.330 ± nan | 0.801 ± nan | 0.616 ± nan |
| GraphSetConv-CrossAttn | 128 | 900,742 | 1.272 ± nan | 0.864 ± nan | 0.649 ± nan |
| GCN+DeepSets | 128 | 154,498 | 1.132 ± nan | 0.784 ± nan | 0.722 ± nan |
| GCN+SetTransformer | 128 | 287,362 | 1.074 ± nan | 0.720 ± nan | **0.750 ± nan** |

## Pairwise Wilcoxon on RMSE (Holm-Bonferroni adjusted)

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|
