# BDE benchmark

- Models trained: 4
- Seeds: 1
- Epochs (max): 5, patience: 10
- Equalization mode: `smaller`
- Hidden (default): 128, Layers: 6, max_conformers: 4

## Test MAE on binding energy (lower is better)

| model | hidden | params | test MAE |
|---|---:|---:|---|
| GraphSetConv-PaiNN-Broadcast | 96 | 3,020,365 | 40.5690 ± nan |
| GraphSetConv-PaiNN-CrossAttn | 92 | 3,191,861 | 23.9043 ± nan |
| PaiNN+DeepSets | 128 | 3,059,969 | 14.1142 ± nan |
| PaiNN+SetTransformer | 128 | 3,325,697 | **8.0927 ± nan** |

## Pairwise Wilcoxon signed-rank tests (Holm-Bonferroni adjusted)

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|
