# eval_molnet_subgraph: bace (classification, random split)

## Per-condition test metrics (mean ± 95% CI; higher AUROC better)

| cond | name | params | test AUROC | test acc |
|---|---|---:|---|---|
| A | GST single | 44,132 | 0.9184 ± 0.0502 | 0.8355 ± 0.0654 |
| E | GCN baseline | 19,649 | 0.9073 ± 0.0160 | 0.8311 ± 0.0340 |
| I | GCN+DS + node-del subgraphs | 27,969 | 0.8752 ± 0.0376 | 0.8092 ± 0.0283 |
| H | GST + node-del subgraphs | 44,132 | 0.9158 ± 0.0388 | 0.8487 ± 0.0327 |

## Paired comparisons on test AUROC (positive d_z = first model better)

| comparison | mean Δ AUROC (a−b) | Cohen's d_z |
|---|---:|---:|
| I vs E | -0.0321 | -1.87 |
| H vs I | +0.0406 | +5.61 |
| H vs A | -0.0026 | -0.07 |
| H vs E | +0.0085 | +0.42 |
| A vs E | +0.0111 | +0.67 |
