# eval_molnet_subgraph: bace (classification, random split)

## Per-condition test metrics (mean ± 95% CI; higher AUROC better)

| cond | name | params | test AUROC | test acc |
|---|---|---:|---|---|
| A | GST single | 44,132 | 0.9073 ± nan | 0.8487 ± nan |
| E | GCN baseline | 19,649 | 0.9222 ± nan | 0.8421 ± nan |
| I | GCN+DS + node-del subgraphs | 27,969 | 0.8744 ± nan | 0.7895 ± nan |
| H | GST + node-del subgraphs | 44,132 | 0.9139 ± nan | 0.8487 ± nan |

## Paired comparisons on test AUROC (positive d_z = first model better)

| comparison | mean Δ AUROC (a−b) | Cohen's d_z |
|---|---:|---:|
| I vs E | -0.0477 | +nan |
| H vs I | +0.0394 | +nan |
| H vs A | +0.0065 | +nan |
| H vs E | -0.0083 | +nan |
| A vs E | -0.0149 | +nan |
