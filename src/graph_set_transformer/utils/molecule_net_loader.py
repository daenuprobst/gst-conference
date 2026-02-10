from pathlib import Path
import numpy as np
import pandas as pd
import deepchem.molnet as mn

from graph_set_transformer.data import make_label_homogeneous_sets
from .scaffold_split import scaffold_split
from graph_set_transformer.utils.graph_encoder import GraphEncoder

MOLECULENET_TASKS = {
    "bace": ["Class"],
    "bbbp": ["p_np"],
    "clintox": ["FDA_APPROVED", "CT_TOX"],
    "esol": ["ESOL predicted log solubility in mols per litre"],
    "freesolv": ["expt"],
    "hiv": ["HIV_active"],
    "lipo": ["exp"],
    "muv": [
        "MUV-692",
        "MUV-689",
        "MUV-846",
        "MUV-859",
        "MUV-644",
        "MUV-548",
        "MUV-852",
        "MUV-600",
        "MUV-810",
        "MUV-712",
        "MUV-737",
        "MUV-858",
        "MUV-713",
        "MUV-733",
        "MUV-652",
        "MUV-466",
        "MUV-832",
    ],
    "qm7": ["u0_atom"],
    "qm8": [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ],
    "qm9": ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv"],
    "sider": [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    "tox21": [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
}


def molecule_net_task_loader(name: str, featurizer=None, **kwargs):
    return MOLECULENET_TASKS[name]


def from_df(df, smiles_column, y_columns):
    return (
        df[smiles_column].to_numpy(),
        df[df.columns.intersection(y_columns)].to_numpy(),
    )


def molecule_net_loader(
    name,
    path,
    task_idx=0,
    featurizer=None,
    split_ratio=0.7,
    seed=42,
    task_name=None,
    **kwargs,
):
    enc = GraphEncoder()

    df = pd.read_csv(path)

    # Drop NAs. Needed in Tox21
    if name in ["tox21"]:
        df = df.replace("", np.nan)
        df = df.dropna(subset=[task_name])

    train_ids, valid_ids, test_ids = scaffold_split(df, 0.1, 0.1, seed)

    train = df.loc[train_ids]
    valid = df.loc[valid_ids]
    test = df.loc[test_ids]

    tasks = MOLECULENET_TASKS[name]

    train_smiles, train_y = from_df(train, "smiles", tasks)
    valid_smiles, valid_y = from_df(valid, "smiles", tasks)
    test_smiles, test_y = from_df(test, "smiles", tasks)

    if len(train_y.shape) == 1:
        train_y = np.expand_dims(train_y, -1)
        valid_y = np.expand_dims(valid_y, -1)
        test_y = np.expand_dims(test_y, -1)

    train_y = np.array(train_y[:, task_idx])
    valid_y = np.array(valid_y[:, task_idx])
    test_y = np.array(test_y[:, task_idx])

    print("Encoding training set ...")
    train_dataset = enc.encode(train_smiles, train_y)
    print("Encoding validation set ...")
    valid_dataset = enc.encode(valid_smiles, valid_y)
    print("Encoding test set ...")
    test_dataset = enc.encode(test_smiles, test_y)

    return train_dataset, valid_dataset, test_dataset, tasks


def get_class_weights(y, task_idx=None):
    if task_idx is None:
        _, counts = np.unique(y, return_counts=True)
        weights = [1 - c / y.shape[0] for c in counts]

        return np.array(weights), np.array(counts)
    else:
        y_t = y.T

        _, counts = np.unique(y_t[task_idx], return_counts=True)
        weights = [1 - c / y_t[task_idx].shape[0] for c in counts]

        return np.array(weights), np.array(counts)
