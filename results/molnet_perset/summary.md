# molnet_perset benchmark

- Datasets: esol
- K sweep: [5, 10]
- Seeds: 1
- Hidden: 128, Layers: 3
- Equalize params: `none`
- Split: `scaffold` (train_ratio=0.8, val_ratio=0.1)
- Feature mode: `rich`
- Epochs (max): 40, patience: 10

## esol

| model | K=5 (rmse) | K=10 (rmse) |
|---|---|---|
| GCN+DeepSets | 6.688 ± nan | 6.646 ± nan |
| GCN+SetTransformer | 7.105 ± nan | 6.749 ± nan |
| GraphSetConv-Broadcast | 6.773 ± nan | 7.243 ± nan |
| GraphSetConv-CrossAttn | 7.305 ± nan | 7.289 ± nan |

| model | K=5 (r) | K=10 (r) |
|---|---|---|
| GCN+DeepSets | 0.776 ± nan | 0.688 ± nan |
| GCN+SetTransformer | 0.516 ± nan | 0.466 ± nan |
| GraphSetConv-Broadcast | 0.815 ± nan | 0.783 ± nan |
| GraphSetConv-CrossAttn | 0.777 ± nan | 0.750 ± nan |

### Set-context benefit (K=5 → K=10 test RMSE delta)

| model | K=5 | K=10 | Δ (lower is better) |
|---|---|---|---|
| GCN+DeepSets | 6.688 | 6.646 | ↓+0.042 |
| GCN+SetTransformer | 7.105 | 6.749 | ↓+0.356 |
| GraphSetConv-Broadcast | 6.773 | 7.243 | ↑+0.471 |
| GraphSetConv-CrossAttn | 7.305 | 7.289 | ↓+0.017 |
