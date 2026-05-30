# DREAM Olfactory Mixtures — GSC vs pipelines (Siamese vs Joint)

- Models: gsc-bc gsc-ca gcn-ds gcn-st
- Framings: siamese joint
- Seeds: 1
- Epochs (max): 30, patience: 5
- Hidden: 32, Layers: 2
- Equalize params: `smaller`
- Side identity (joint mode): `on`
- Feature mode: `rich`
- Splits: train_ratio=0.85, val_ratio=0.1, split_seed=0

## Test metrics (mean ± 95% CI)

| model | framing | params | RMSE | Pearson r |
|---|---|---:|---|---|
| GraphSetConv-Broadcast | siamese | 14,500 | 0.1216 ± nan | **0.5668 ± nan** |
| GraphSetConv-Broadcast | joint | 14,532 | 0.1359 ± nan | 0.4603 ± nan |
| GraphSetConv-CrossAttn | siamese | 16,836 | 0.1388 ± nan | 0.4220 ± nan |
| GraphSetConv-CrossAttn | joint | 16,868 | 0.1484 ± nan | 0.1967 ± nan |
| GCN+DeepSets | siamese | 12,866 | 0.1500 ± nan | 0.0053 ± nan |
| GCN+DeepSets | joint | 12,930 | 0.1386 ± nan | 0.4102 ± nan |
| GCN+SetTransformer | siamese | 21,506 | 0.1410 ± nan | 0.3547 ± nan |
| GCN+SetTransformer | joint | 21,570 | 0.1391 ± nan | 0.3337 ± nan |

## Framing deltas (joint − siamese)

Per-architecture, mean (joint Pearson) − mean (siamese Pearson). POSITIVE values = the joint framing helps that architecture. GSC's specific architectural advantage is that it should benefit MORE from joint than the pipelines do.

| model | Δ Pearson (joint − siamese) | Δ RMSE | paired d_z (Pearson) |
|---|---:|---:|---:|
| GraphSetConv-Broadcast | -0.1065 | +0.0142 | n/a |
| GraphSetConv-CrossAttn | -0.2253 | +0.0096 | n/a |
| GCN+DeepSets | +0.4050 | -0.0113 | n/a |
| GCN+SetTransformer | -0.0210 | -0.0019 | n/a |

