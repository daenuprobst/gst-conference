# Peptides (func) benchmark

- Task: `func`  (10-class multilabel AP / 11-target regression MAE)
- Partition: METIS, K = 8
- Models: 4
- Seeds: 1
- Epochs (max): 3, patience: 3
- Equalization mode: `none`
- Hidden (default): 96, Layers: 4

## Test metrics (mean ± 95% CI)

| model | hidden | params | ap | bce_loss |
|---|---|---|---|---|
| GraphSetConv-Broadcast | 96 | 513,518 | 0.3316 ± nan | 0.3428 ± nan |
| GraphSetConv-CrossAttn | 96 | 663,278 | **0.3689 ± nan** | 0.3384 ± nan |
| GCN+DeepSets | 96 | 102,250 | 0.2232 ± nan | 0.3608 ± nan |
| GCN+SetTransformer | 96 | 177,322 | 0.2471 ± nan | 0.3551 ± nan |

## Pairwise Wilcoxon on AP (Holm-Bonferroni adjusted)

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|
