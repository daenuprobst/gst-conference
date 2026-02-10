import pickle
import numpy as np
import pandas as pd
from tdc.single_pred import ADME
from tdc.multi_pred import DDI

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rdkit.Chem.AllChem import MolFromSmiles
from graph_set_transformer.utils.graph_encoder import GraphEncoder


def from_df(df, smiles_column, y_columns):
    return (
        df[smiles_column].to_numpy(),
        df[df.columns.intersection(y_columns)].to_numpy(),
    )


def load_reaction_data(csv_path):
    df = pd.read_csv(csv_path)

    df = pd.read_csv(csv_path)
    enc = GraphEncoder()

    rxns = []

    for label, reaction_smiles in tqdm(
        zip(df["class"], df["reactants>reagents>production"]), total=len(df)
    ):
        parts = str(reaction_smiles).split(">")

        all_molecules = []

        for part in parts:
            part_molecules = [mol for mol in part.split(".") if mol]
            all_molecules.extend(part_molecules)

        data = enc.encode(
            all_molecules, [label - 1] * len(all_molecules), disable_tqdm=True
        )

        rxns.append((data, label - 1))

    return rxns


def load_ddi_data(csv_path, frac=1.0):
    with open(csv_path, "rb") as f:
        df = pickle.load(f)

    print(df["Y"].max())

    df = df.sample(n=int(len(df) * frac), random_state=42)
    enc = GraphEncoder()

    pairs = []

    for label, drug_1, drug_2 in tqdm(
        zip(df["Y"], df["Drug1"], df["Drug2"]), total=len(df)
    ):
        mol_1 = MolFromSmiles(drug_1)
        mol_2 = MolFromSmiles(drug_2)

        if mol_1 is None or mol_2 is None:
            print("Unreadable SMILES ...")
            continue

        all_molecules = [drug_1, drug_2]

        data = enc.encode(
            all_molecules, [label - 1] * len(all_molecules), disable_tqdm=True
        )

        pairs.append((data, label - 1))

    return pairs


def tdc_adme_task_loader(name: str, featurizer=None, **kwargs):
    # return ["Solubility_AqSolDB"]
    # return ["Lipophilicity_AstraZeneca"]
    # return ["PPBR_AZ"]
    # return ["Bioavailability_Ma"]
    # return ["CYP2C9_Veith"]
    return ["BBB_Martins"]


def tdc_adme_loader(name: str, featurizer=None, seed=42, **kwargs):
    enc = GraphEncoder()

    # task_name = kwargs.get("task_name", None)
    # print(task_name)

    data = ADME(name=name)
    split = data.get_split(method="scaffold")

    tasks = ["Y"]

    train_smiles, train_y = from_df(split["train"], "Drug", tasks)
    valid_smiles, valid_y = from_df(split["valid"], "Drug", tasks)
    test_smiles, test_y = from_df(split["test"], "Drug", tasks)

    if len(train_y.shape) == 1:
        train_y = np.expand_dims(train_y, -1)
        valid_y = np.expand_dims(valid_y, -1)
        test_y = np.expand_dims(test_y, -1)

    train_y = np.array(train_y[:, 0])
    valid_y = np.array(valid_y[:, 0])
    test_y = np.array(test_y[:, 0])

    print("Encoding training set ...")
    train_dataset = enc.encode(train_smiles, train_y)
    print("Encoding validation set ...")
    valid_dataset = enc.encode(valid_smiles, valid_y)
    print("Encoding test set ...")
    test_dataset = enc.encode(test_smiles, test_y)

    return train_dataset, valid_dataset, test_dataset, tasks


def tdc_ddi_loader(
    path_train, path_valid, path_test, featurizer=None, seed=42, **kwargs
):
    train = load_ddi_data(path_train, 0.01)
    val = load_ddi_data(path_valid, 0.1)
    test = load_ddi_data(path_test, 0.1)

    return train, val, test
