# Graph Set Transformer (GST)

## Dev instructions

- Install uv ([instructions](https://docs.astral.sh/uv/getting-started/installation/))
- Run `uv sync`
- To run scripts use e.g. `uv run scripts/train.py`

## Usage

### Dataset-Specific Training

Train on specific molecular datasets:

```bash
# Drug-Drug Interaction prediction
python scripts/train_drug_drug.py

# MoleculeNet benchmarks
python scripts/train_moleculenet.py
```

### Custom Training

```python
from graph_set_transformer.models import GraphSetTransformerGraphClassifier
from graph_set_transformer.data import SetDataset, make_label_homogeneous_sets

# Initialize model
model = GraphSetTransformerGraphClassifier(
    in_channels=num_features,
    hidden_dim=64,
    num_classes=2
)

# Prepare set-based data
sets = make_label_homogeneous_sets(dataset, set_size=10)
set_dataset = SetDataset(sets)

# Train your model...
```

Built with:

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [RDKit](https://www.rdkit.org/)
- [DeepChem](https://deepchem.io/)
