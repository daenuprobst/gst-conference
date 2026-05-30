# Graph Set Transformer (GST)

Most graph neural networks look at one graph at a time. The Graph Set Transformer
looks at a *set* of graphs together and lets information flow between them while
it reasons.

The idea lives in a single building block, `GraphSetConv`. Inside each layer the
nodes of every graph still pass messages along their own edges, but they also
attend to a shared set of tokens summarising the whole set. So a node's
representation can depend on what the other graphs in the set look like, not just
its own neighbourhood.

This matters for problems where the answer for one graph genuinely depends on its
companions. A chemical reaction is the clearest example: the reactive site of a
molecule changes depending on what else is in the flask.

## Why a set?

A lot of tasks are naturally about a group of graphs rather than one:

- **Reactions** — a set of reactant molecules, and you want to know which atoms
  react or what the yield will be.
- **Mixtures and formulations** — several components whose joint behaviour is the
  thing you care about.
- **Any case where context matters** — the same graph can have different labels
  depending on the company it keeps.

The usual approach is to encode each graph on its own and then pool the results
(DeepSets, Set Transformer, and friends). That keeps the graphs blind to each
other until the very end. GST instead mixes set-level information in early and
often, which is the whole point of the experiments below.

## Install

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

That sets up Python 3.13+ and everything in `pyproject.toml` (PyTorch, PyTorch
Geometric, RDKit, DeepChem, and the data tooling).

## Repository layout

```
src/graph_set_transformer/
    models/      the GraphSetConv block and the 2D / 3D (PaiNN) variants
    data/        SetDataset and helpers for building sets of graphs
    utils/       graph encoders, dataset loaders, scaffold splitting
scripts/         benchmark runners and dataset builders
data/            cached datasets used by the benchmarks
results/         output from benchmark runs
```

## Running the benchmarks

Every experiment is a standalone script in `scripts/`. They all compare the same
four models under a matched parameter budget, so you can see where set-level
reasoning actually helps:

- **GraphSetConv (cross-attention)** — nodes attend to per-set tokens
- **GraphSetConv (broadcast)** — a simpler uniform set context
- **GCN + DeepSets** — encode each graph, then pool
- **GCN + Set Transformer** — encode each graph, then attend over graphs

Run any of them from the repo root:

```bash
# Synthetic diagnostics (set-conditional distance, structural reasoning)
python scripts/eval_synthetic.py

# Reaction-center prediction on USPTO-15K (per-atom labels)
python scripts/eval_rxn_center.py

# Buchwald-Hartwig reaction-yield prediction
python scripts/eval_react.py

# Over-squashing stress test on synthetic dumbbell graphs
python scripts/eval_oversquash.py

# Atom mapping on USPTO-50K
python scripts/atom_mapping.py
```

Most scripts take a `--quick` flag for a fast sanity run and write their output
under `results/`. Pass `-h` to any of them to see the available options. The
`*_data.py` scripts build or cache the datasets the benchmarks expect.

## Using the model directly

The set machinery is just an ordinary PyTorch module. A `SetDataset` turns a list
of graphs into batches of graph-sets:

```python
from graph_set_transformer.data import SetDataset, make_label_homogeneous_sets

sets = make_label_homogeneous_sets(dataset, set_size=10)
set_dataset = SetDataset(sets)
```

The model definitions live under `graph_set_transformer.models` — see the
benchmark scripts for end-to-end examples of wiring a model, dataset, and
training loop together.

## Built with

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [RDKit](https://www.rdkit.org/)
- [DeepChem](https://deepchem.io/)
