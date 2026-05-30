# eval_setaug_synth results

## Per-condition test metrics (mean ± 95% CI over seeds)

| cond | name | params | test acc | test F1 (macro) | best val acc |
|---|---|---:|---|---|---|
| A | GST single | 43,335 | 0.2150 ± nan | 0.1459 ± nan | 0.2000 |
| B | GST set-aug | 43,335 | 0.2800 ± nan | 0.1196 ± nan | 0.2400 |
| C | GCN+SupCon | 8,100 | 0.2900 ± nan | 0.1940 ± nan | 0.2600 |
| D | GCN+DS set-aug | 7,044 | 0.2600 ± nan | 0.1032 ± nan | 0.3200 |
| E | GCN baseline | 4,932 | 0.2200 ± nan | 0.1323 ± nan | 0.2100 |

## Paired comparisons on test accuracy (Holm-adjusted)

| comparison | mean diff | Cohen's d_z | p (adj) | reject H0 |
|---|---:|---:|---:|:---:|
| B vs A | +0.0650 | +nan | 1.000 | no |
| B vs C | -0.0100 | +nan | 1.000 | no |
| B vs D | +0.0200 | +nan | 1.000 | no |
| B vs E | +0.0600 | +nan | 1.000 | no |
| D vs E | +0.0400 | +nan | 1.000 | no |
| C vs E | +0.0700 | +nan | 1.000 | no |
| A vs E | -0.0050 | +nan | 1.000 | no |
