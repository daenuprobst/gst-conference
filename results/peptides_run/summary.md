# Peptides (func) benchmark

- Task: `func`  (10-class multilabel AP / 11-target regression MAE)
- Partition: METIS, K = 8
- Models: 4
- Seeds: 1
- Epochs (max): 10, patience: 3
- Equalization mode: `smaller`
- Hidden (default): 96, Layers: 4

## Test metrics (mean ± 95% CI)

| model | hidden | params | ap | bce_loss |
|---|---|---|---|---|
| GraphSetConv-Broadcast | 40 | 95,574 | 0.3888 ± nan | 0.3102 ± nan |
| GraphSetConv-CrossAttn | 36 | 99,986 | **0.4150 ± nan** | 0.3039 ± nan |
| GCN+DeepSets | 96 | 102,250 | 0.2261 ± nan | 0.3545 ± nan |
| GCN+SetTransformer | 96 | 177,322 | 0.2346 ± nan | 0.3513 ± nan |

## Pairwise Wilcoxon on AP (Holm-Bonferroni adjusted)

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|
