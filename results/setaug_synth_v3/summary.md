# eval_setaug_synth results

## Per-condition test metrics (mean ± 95% CI over seeds)

| cond | name | params | test acc | test F1 (macro) | best val acc |
|---|---|---:|---|---|---|
| A | GST single | 43,335 | 0.4020 ± 0.0124 | 0.3720 ± 0.0301 | 0.4167 |
| E | GCN baseline | 4,932 | 0.3620 ± 0.0043 | 0.3258 ± 0.0507 | 0.3622 |
| H | GST + node-del subgraphs | 43,335 | 0.4390 ± 0.0358 | 0.4127 ± 0.0312 | 0.4333 |
| I | GCN+DS + node-del subgraphs | 7,044 | 0.3980 ± 0.0400 | 0.3467 ± 0.0717 | 0.3978 |

## Paired comparisons on test accuracy (Holm-adjusted)

| comparison | mean diff | Cohen's d_z | p (adj) | reject H0 |
|---|---:|---:|---:|:---:|
| A vs E | +0.0400 | +6.10 | 1.000 | no |
| H vs A | +0.0370 | +2.03 | 1.000 | no |
| H vs E | +0.0770 | +6.03 | 1.000 | no |
| H vs I | +0.0410 | +2.40 | 1.000 | no |
| I vs E | +0.0360 | +2.25 | 1.000 | no |
