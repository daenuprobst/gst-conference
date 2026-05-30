# eval_molnet_subgraph: bace (classification, random split)

## Per-condition test metrics (mean ± 95% CI; higher AUROC better)

| cond | name | params | test AUROC | test acc |
|---|---|---:|---|---|
| I | GCN+DS + node-del subgraphs | 27,969 | 0.8690 ± 0.0400 | 0.8224 ± 0.0432 |
| H | GST + node-del subgraphs | 44,132 | 0.9060 ± 0.0293 | 0.8333 ± 0.0377 |

## Paired comparisons on test AUROC (positive d_z = first model better)

| comparison | mean Δ AUROC (a−b) | Cohen's d_z |
|---|---:|---:|
| H vs I | +0.0371 | +1.46 |
