# eval_molnet_subgraph: bace (classification, random split)

## Per-condition test metrics (mean ± 95% CI; higher AUROC better)

| cond | name | params | test AUROC | test acc |
|---|---|---:|---|---|
| A | GST single | 44,132 | 0.9139 ± 0.0255 | 0.8421 ± 0.0432 |
| E | GCN baseline | 19,649 | 0.9179 ± 0.0267 | 0.8399 ± 0.0094 |
| I | GCN+DS + node-del subgraphs | 27,969 | 0.8828 ± 0.0342 | 0.8048 ± 0.0525 |
| H | GST + node-del subgraphs | 44,132 | 0.9142 ± 0.0073 | 0.8377 ± 0.0737 |

## Paired comparisons on test AUROC (positive d_z = first model better)

| comparison | mean Δ AUROC (a−b) | Cohen's d_z |
|---|---:|---:|
| I vs E | -0.0352 | -1.44 |
| H vs I | +0.0314 | +2.30 |
| H vs A | +0.0002 | +0.03 |
| H vs E | -0.0038 | -0.34 |
| A vs E | -0.0040 | -0.22 |
