# USPTO-15K reaction-centre prediction

- Run: `results/uspto15k_run`
- Seeds: 5
- Epochs: 20
- Equalisation mode: `smaller`

## Dataset audit

| split | reactions | mean #reactants/rxn | mean #atoms/rxn | positive density (atoms) |
|---|---:|---:|---:|---:|
| train | 1000 | 3.57 | 34.8 | 0.0949 |
| valid | 1495 | 3.59 | 34.5 | 0.0975 |
| test | 2989 | 3.61 | 34.9 | 0.0959 |

## Parameter budgets

| model | hidden | parameters |
|---|---:|---:|
| GraphSetConv-CrossAttn | 76 | 130,798 |
| GraphSetConv-Broadcast | 84 | 130,622 |
| GCN+DeepSets | 128 | 134,913 |
| GCN+SetTransformer | 128 | 267,777 |

## Test AP (mean ± 95% Student-t CI half-width over seeds)

| model | AP | n |
|---|---:|---:|
| GraphSetConv-CrossAttn | 0.6089 ± 0.0144 | 5 |
| GraphSetConv-Broadcast | 0.6195 ± 0.0059 | 5 |
| GCN+DeepSets | 0.6125 ± 0.0035 | 5 |
| GCN+SetTransformer | 0.6158 ± 0.0047 | 5 |

## Pairwise Wilcoxon signed-rank (Holm-Bonferroni adjusted)

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|
| GraphSetConv-CrossAttn vs GraphSetConv-Broadcast | 0.75 | -0.73 | no |
| GraphSetConv-CrossAttn vs GCN+DeepSets | 0.875 | -0.29 | no |
| GraphSetConv-CrossAttn vs GCN+SetTransformer | 0.875 | -0.52 | no |
| GraphSetConv-Broadcast vs GCN+DeepSets | 0.375 | +2.63 | no |
| GraphSetConv-Broadcast vs GCN+SetTransformer | 0.75 | +0.74 | no |
| GCN+DeepSets vs GCN+SetTransformer | 0.625 | -1.08 | no |
