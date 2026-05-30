# eval_setaug_synth results

## Per-condition test metrics (mean ± 95% CI over seeds)

| cond | name | params | test acc | test F1 (macro) | best val acc |
|---|---|---:|---|---|---|
| F | GST set-aug (var k 1..K) | 43,335 | 0.2600 ± nan | 0.2486 ± nan | 0.2500 |
| G | GCN+DS set-aug (var k 1..K) | 7,044 | 0.2600 ± nan | 0.1032 ± nan | 0.3200 |

## Paired comparisons on test accuracy (Holm-adjusted)

| comparison | mean diff | Cohen's d_z | p (adj) | reject H0 |
|---|---:|---:|---:|:---:|
| F vs G | +0.0000 | +nan | 1.000 | no |
