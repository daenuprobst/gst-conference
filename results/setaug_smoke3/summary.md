# eval_setaug_synth results

## Per-condition test metrics (mean ± 95% CI over seeds)

| cond | name | params | test acc | test F1 (macro) | best val acc |
|---|---|---:|---|---|---|
| H | GST + node-del subgraphs | 43,335 | 0.2700 ± nan | 0.1063 ± nan | 0.2400 |
| I | GCN+DS + node-del subgraphs | 7,044 | 0.3250 ± nan | 0.2111 ± nan | 0.3100 |

## Paired comparisons on test accuracy (Holm-adjusted)

| comparison | mean diff | Cohen's d_z | p (adj) | reject H0 |
|---|---:|---:|---:|:---:|
| H vs I | -0.0550 | +nan | 1.000 | no |
