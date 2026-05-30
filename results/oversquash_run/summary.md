# Over-squashing benchmark

- Sweep axis: `B`
- Models: 5
- Seeds: 1
- Hidden: 64, Layers: 3
- Train/val/test = 200/50/100

## Config: {'K': 2, 'N': 20, 'B': 1}

| model | params | RMSE | MAE | R² |
|---|---:|---|---|---|
| GCN-only (uncut) | 17,409 | 0.0174 ± nan | 0.0131 ± nan | -0.0080 ± nan |
| GCN+DeepSets (cut) | 34,049 | 0.0174 ± nan | 0.0130 ± nan | -0.0099 ± nan |
| GCN+SetTransformer (cut) | 67,713 | 0.0174 ± nan | 0.0132 ± nan | -0.0067 ± nan |
| GraphSetConv-V1 (cut) | 168,324 | 0.0174 ± nan | 0.0130 ± nan | -0.0052 ± nan |
| GraphSetConv-V2 (cut) | 218,628 | **0.0174 ± nan** | 0.0131 ± nan | -0.0047 ± nan |

## Config: {'K': 2, 'N': 20, 'B': 8}

| model | params | RMSE | MAE | R² |
|---|---:|---|---|---|
| GCN-only (uncut) | 17,409 | 0.0228 ± nan | 0.0175 ± nan | -0.0375 ± nan |
| GCN+DeepSets (cut) | 34,049 | 0.0225 ± nan | 0.0171 ± nan | -0.0075 ± nan |
| GCN+SetTransformer (cut) | 67,713 | 0.0225 ± nan | 0.0173 ± nan | -0.0094 ± nan |
| GraphSetConv-V1 (cut) | 168,324 | **0.0224 ± nan** | 0.0171 ± nan | -0.0044 ± nan |
| GraphSetConv-V2 (cut) | 218,628 | 0.0225 ± nan | 0.0172 ± nan | -0.0141 ± nan |

