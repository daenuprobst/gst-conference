# eval_molnet_subgraph: freesolv (scaffold split)

## Per-condition test metrics (mean ± 95% CI over seeds; lower RMSE better)

| cond | name | params | test RMSE | test MAE | test R² |
|---|---|---:|---|---|---|
| A | GST single | 170,180 | 2.8591 ± 0.8537 | 1.9444 ± 0.3234 | 0.6300 ± 0.2139 |
| E | GCN baseline | 19,649 | 2.1585 ± 0.1688 | 1.4339 ± 0.0653 | 0.7910 ± 0.0325 |
| I | GCN+DS + node-del subgraphs | 27,969 | 2.1421 ± 0.4293 | 1.4538 ± 0.3783 | 0.7934 ± 0.0829 |
| H | GST + node-del subgraphs | 170,180 | 2.3654 ± 0.3591 | 1.6945 ± 0.2058 | 0.7486 ± 0.0751 |

## Paired comparisons on test RMSE (positive d_z = first model better)

| comparison | mean Δ RMSE (b−a) | Cohen's d_z |
|---|---:|---:|
| I vs E | +0.0164 | +0.07 |
| H vs I | -0.2233 | -0.94 |
| H vs A | +0.4938 | +1.12 |
| H vs E | -0.2069 | -1.30 |
| A vs E | -0.7006 | -2.41 |
