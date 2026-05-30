# Polymer EA/IP benchmark

- Models: 5
- Seeds: 1
- Epochs (max): 200, patience: 30
- Equalization mode: `smaller`
- Hidden (default): 128, Layers: 3
- Targets: EA, IP (eV)

## Test metrics — overall (mean across targets, ± 95% CI over seeds)

| model | hidden | params | RMSE (eV) | MAE (eV) | R² |
|---|---:|---:|---|---|---|
| GraphSetConv-V1 | 56 | 131,157 | 0.1668 ± nan | 0.1176 ± nan | 0.8910 ± nan |
| GraphSetConv-V2-CrossAttn | 52 | 146,853 | 0.1550 ± nan | 0.1056 ± nan | 0.9058 ± nan |
| GraphSetConv-V3 | 44 | 137,238 | 0.2059 ± nan | 0.1254 ± nan | 0.8363 ± nan |
| GCN+DeepSets | 128 | 138,114 | 0.1654 ± nan | 0.1131 ± nan | 0.8911 ± nan |
| GCN+SetTransformer | 128 | 270,978 | 0.1484 ± nan | 0.1046 ± nan | **0.9118 ± nan** |

## Per-target test metrics

### EA (eV)

| model | RMSE | MAE | R² |
|---|---|---|---|
| GraphSetConv-V1 | 0.1628 ± nan | 0.1214 ± nan | 0.8802 ± nan |
| GraphSetConv-V2-CrossAttn | 0.1516 ± nan | 0.1118 ± nan | 0.8961 ± nan |
| GraphSetConv-V3 | 0.1770 ± nan | 0.1383 ± nan | 0.8583 ± nan |
| GCN+DeepSets | 0.1700 ± nan | 0.1230 ± nan | 0.8693 ± nan |
| GCN+SetTransformer | 0.1549 ± nan | 0.1180 ± nan | 0.8915 ± nan |

### IP (eV)

| model | RMSE | MAE | R² |
|---|---|---|---|
| GraphSetConv-V1 | 0.1707 ± nan | 0.1138 ± nan | 0.9019 ± nan |
| GraphSetConv-V2-CrossAttn | 0.1584 ± nan | 0.0993 ± nan | 0.9155 ± nan |
| GraphSetConv-V3 | 0.2349 ± nan | 0.1125 ± nan | 0.8142 ± nan |
| GCN+DeepSets | 0.1608 ± nan | 0.1032 ± nan | 0.9130 ± nan |
| GCN+SetTransformer | 0.1419 ± nan | 0.0913 ± nan | 0.9322 ± nan |

## Pairwise Wilcoxon on overall RMSE (Holm-Bonferroni adjusted)

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|
