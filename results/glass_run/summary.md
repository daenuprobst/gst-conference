# GLASS / ppi_bp — set-of-subgraphs benchmark

- Dataset: `ppi_bp`  (Accuracy)
- K values: [1, 4, 8]
- Seeds per K: 1
- Steps: 800, eval every 100
- Equalization mode: `smaller`
- Hidden (default): 64, Layers: 4, emb_dim: 64

## Per-K test results (mean ± 95% CI)

### K = 1

| model | hidden | non-emb params | Accuracy |
|---|---:|---:|---|
| GraphSetConv-Broadcast | 16 | 26,875 | 0.4625 ± nan |
| GraphSetConv-CrossAttn | 16 | 31,547 | **0.4875 ± nan** |
| GCN+DeepSets | 64 | 46,983 | 0.2562 ± nan |
| GCN+SetTransformer | 64 | 63,815 | 0.3312 ± nan |

**Pairwise Wilcoxon (Holm-Bonferroni adjusted):**

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|

### K = 4

| model | hidden | non-emb params | Accuracy |
|---|---:|---:|---|
| GraphSetConv-Broadcast | 16 | 26,875 | **0.5312 ± nan** |
| GraphSetConv-CrossAttn | 16 | 31,547 | 0.4875 ± nan |
| GCN+DeepSets | 64 | 46,983 | 0.2375 ± nan |
| GCN+SetTransformer | 64 | 63,815 | 0.2875 ± nan |

**Pairwise Wilcoxon (Holm-Bonferroni adjusted):**

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|

### K = 8

| model | hidden | non-emb params | Accuracy |
|---|---:|---:|---|
| GraphSetConv-Broadcast | 16 | 26,875 | 0.5125 ± nan |
| GraphSetConv-CrossAttn | 16 | 31,547 | **0.5188 ± nan** |
| GCN+DeepSets | 64 | 46,983 | 0.3063 ± nan |
| GCN+SetTransformer | 64 | 63,815 | 0.3250 ± nan |

**Pairwise Wilcoxon (Holm-Bonferroni adjusted):**

| comparison | p (adj) | Cohen's d_z | reject H0 |
|---|---:|---:|:---:|

