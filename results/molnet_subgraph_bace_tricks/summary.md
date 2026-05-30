# eval_molnet_subgraph: bace (classification, random split)

## Per-condition test metrics (mean ± 95% CI; higher AUROC better)

| cond | name | params | test AUROC | test acc |
|---|---|---:|---|---|
| A | GST single | 44,132 | 0.9268 ± 0.0303 | 0.8465 ± 0.0250 |
| E | GCN baseline | 19,649 | 0.8891 ± 0.0295 | 0.8509 ± 0.0340 |
| I | GCN+DS + node-del subgraphs | 27,969 | 0.8833 ± 0.0411 | 0.8268 ± 0.0499 |
| H | GST + node-del subgraphs | 44,132 | 0.9234 ± 0.0122 | 0.8531 ± 0.0094 |

## Paired comparisons on test AUROC (positive d_z = first model better)

| comparison | mean Δ AUROC (a−b) | Cohen's d_z |
|---|---:|---:|
| I vs E | -0.0058 | -0.96 |
| H vs I | +0.0401 | +3.10 |
| H vs A | -0.0034 | -0.44 |
| H vs E | +0.0344 | +3.53 |
| A vs E | +0.0377 | +4.90 |
