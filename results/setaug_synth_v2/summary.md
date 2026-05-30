# eval_setaug_synth results

## Per-condition test metrics (mean ± 95% CI over seeds)

| cond | name | params | test acc | test F1 (macro) | best val acc |
|---|---|---:|---|---|---|
| A | GST single | 43,335 | 0.4020 ± 0.0124 | 0.3720 ± 0.0301 | 0.4167 |
| B | GST set-aug (fixed k) | 43,335 | 0.4010 ± 0.0155 | 0.3743 ± 0.0252 | 0.4133 |
| C | GCN+SupCon | 8,100 | 0.3567 ± 0.0080 | 0.3170 ± 0.0702 | 0.3467 |
| D | GCN+DS set-aug (fixed k) | 7,044 | 0.3763 ± 0.0277 | 0.3339 ± 0.0691 | 0.3756 |
| E | GCN baseline | 4,932 | 0.3620 ± 0.0043 | 0.3258 ± 0.0507 | 0.3622 |
| F | GST set-aug (var k 1..K) | 43,335 | 0.4023 ± 0.0349 | 0.3714 ± 0.0397 | 0.3944 |
| G | GCN+DS set-aug (var k 1..K) | 7,044 | 0.3860 ± 0.0489 | 0.3569 ± 0.0210 | 0.3789 |

## Paired comparisons on test accuracy (Holm-adjusted)

| comparison | mean diff | Cohen's d_z | p (adj) | reject H0 |
|---|---:|---:|---:|:---:|
| B vs A | -0.0010 | -0.10 | 1.000 | no |
| B vs C | +0.0443 | +10.65 | 1.000 | no |
| B vs D | +0.0247 | +1.48 | 1.000 | no |
| B vs E | +0.0390 | +6.50 | 1.000 | no |
| D vs E | +0.0143 | +1.34 | 1.000 | no |
| C vs E | -0.0053 | -1.28 | 1.000 | no |
| A vs E | +0.0400 | +6.10 | 1.000 | no |
| F vs A | +0.0003 | +0.02 | 1.000 | no |
| F vs B | +0.0013 | +0.13 | 1.000 | no |
| F vs C | +0.0457 | +3.35 | 1.000 | no |
| F vs E | +0.0403 | +3.19 | 1.000 | no |
| G vs D | +0.0097 | +0.66 | 1.000 | no |
| G vs E | +0.0240 | +1.33 | 1.000 | no |
| F vs G | +0.0163 | +0.97 | 1.000 | no |
