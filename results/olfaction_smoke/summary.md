# DREAM Olfactory Mixtures — Siamese vs Joint (no-provenance)

- Models: gcn-ds gcn-st gsc-bc gsc-ca
- Framings: siamese joint
- Seeds: 1
- Epochs (max): 30, patience: 5
- Hidden: 32, Layers: 2
- Equalize params: `none`
- Feature mode: `rich`
- Splits: train_ratio=0.85, val_ratio=0.1, split_seed=0

Joint mode here is a deliberate ablation: no side embedding,
no per-side readout. The model sees the union of A and B as one
set with no way to recover the partition. Joint should
underperform siamese for all architectures; the (siamese - joint)
gap measures how much each architecture relies on knowing the
partition.

## Test metrics (mean ± 95% CI)

| model | framing | params | RMSE | Pearson r |
|---|---|---:|---|---|
| GCN+DeepSets | siamese | 12,866 | 0.1500 ± nan | 0.0053 ± nan |
| GCN+DeepSets | joint | 9,794 | 0.1504 ± nan | 0.0755 ± nan |
| GCN+SetTransformer | siamese | 21,506 | 0.1410 ± nan | 0.3547 ± nan |
| GCN+SetTransformer | joint | 18,434 | 0.1544 ± nan | -0.1215 ± nan |
| GraphSetConv-Broadcast | siamese | 34,244 | 0.1279 ± nan | **0.5706 ± nan** |
| GraphSetConv-Broadcast | joint | 31,172 | 0.1484 ± nan | -0.1142 ± nan |
| GraphSetConv-CrossAttn | siamese | 42,820 | 0.1527 ± nan | 0.3732 ± nan |
| GraphSetConv-CrossAttn | joint | 39,748 | 0.1491 ± nan | -0.1073 ± nan |

## Framing gaps (siamese − joint)

Per-architecture, mean (siamese Pearson) − mean (joint Pearson). 
POSITIVE = siamese is better, i.e. the architecture relies on 
knowing the A/B partition. SMALL gap = the architecture extracts 
useful signal from set composition alone. LARGE gap = the 
architecture is doing real cross-mixture comparison.

| model | Δ Pearson (siamese − joint) | Δ RMSE | paired d_z (Pearson) |
|---|---:|---:|---:|
| GCN+DeepSets | -0.0703 | -0.0004 | n/a |
| GCN+SetTransformer | +0.4762 | -0.0134 | n/a |
| GraphSetConv-Broadcast | +0.6849 | -0.0205 | n/a |
| GraphSetConv-CrossAttn | +0.4805 | +0.0036 | n/a |

