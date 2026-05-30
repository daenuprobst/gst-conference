# Over-squashing benchmark

- Sweep axis: `B`
- Models: 5
- Seeds: 1
- Hidden: 64, Layers: 3
- Train/val/test = 4000/500/1000

## Config: {'K': 2, 'N': 20, 'B': inf}

| model | params | RMSE | MAE | R² |
|---|---:|---|---|---|
| GCN-only (uncut) | 17,409 | 0.0152 ± nan | 0.0110 ± nan | 0.5186 ± nan |
| GCN+DeepSets (cut) | 34,049 | 0.0220 ± nan | 0.0162 ± nan | -0.0005 ± nan |
| GCN+SetTransformer (cut) | 67,713 | 0.0148 ± nan | 0.0102 ± nan | 0.5448 ± nan |
| GraphSetConv-V1 (cut) | 33,324 | **0.0051 ± nan** | 0.0034 ± nan | 0.9458 ± nan |
| GraphSetConv-V2 (cut) | 32,068 | 0.0064 ± nan | 0.0046 ± nan | 0.9139 ± nan |

