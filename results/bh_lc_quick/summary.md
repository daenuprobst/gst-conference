# BH yield — training-set-size ablation

- Test set: fixed 30% of full dataset (split-seed 0)
- Train fractions: [0.7]
- Seeds per fraction: 1
- Epochs (max): 10, patience: 5
- Feature mode: `minimal`
- Val strategy: `random`
- Min train size for val: 50
- Equalization mode: `smaller`

## Train fraction = 0.700  (n_train ≈ 2718)

| model | RMSE (%) | MAE (%) | R² |
|---|---|---|---|
| GraphSetConv-V1 | 13.972 ± nan | 9.870 ± nan | 0.740 ± nan |
| GraphSetConv-V2-CrossAttn | 13.304 ± nan | 9.166 ± nan | **0.765 ± nan** |
| GCN+DeepSets | 21.552 ± nan | 17.286 ± nan | 0.383 ± nan |
| GCN+SetTransformer | 21.416 ± nan | 16.617 ± nan | 0.390 ± nan |

