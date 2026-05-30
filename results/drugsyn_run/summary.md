# DrugSyn (OncoPolyPharmacology) benchmark

- Dataset: TDC OncoPolyPharmacology (DeepSynergy preprocess)
- Target: Loewe synergy score
- Models: 4
- Seeds: 1
- Epochs (max): 20, patience: 10
- Equalization mode: `none`
- Feature mode: `rich`
- Max samples: 1000
- Hidden (default): 128, Layers: 3
- Cell-line encoder hidden: 128

## Test metrics (mean ± 95% CI)

Reference: DeepSynergy (Preuer et al. 2018, Table 2) — leave-drug-combination-out 5-fold CV: RMSE 15.91, Pearson r 0.73, R² ≈ 0.53. Note: that protocol differs from random row split below.

| model | hidden | params | RMSE | MAE | R² | Pearson r |
|---|---:|---:|---|---|---|---|
| GraphSetConv-Broadcast | 128 | 2,318,980 | 19.231 ± nan | 13.775 ± nan | 0.066 ± nan | 0.383 ± nan |
| GraphSetConv-CrossAttn | 128 | 2,519,044 | 18.734 ± nan | 13.562 ± nan | 0.113 ± nan | 0.400 ± nan |
| GCN+DeepSets | 128 | 1,295,233 | 17.677 ± nan | 12.865 ± nan | **0.211 ± nan** | 0.486 ± nan |
| GCN+SetTransformer | 128 | 1,428,097 | 18.755 ± nan | 13.444 ± nan | 0.111 ± nan | 0.387 ± nan |

## Pairwise Wilcoxon on RMSE (Holm-Bonferroni adjusted)

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|
