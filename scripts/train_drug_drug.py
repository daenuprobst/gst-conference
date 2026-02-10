"""Multi-class classification using set-based models on drug pairs."""

import torch
import torch.nn.functional as F
import numpy as np
import random
import pickle
import sys
import os
import warnings
from collections import defaultdict
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from pathlib import Path

# Suppress sklearn warnings about missing classes in metrics
warnings.filterwarnings("ignore", message=".*No positive class found in y_true.*")
warnings.filterwarnings("ignore", message=".*Only one class is present in y_true.*")


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_set_transformer.models.models import (
    SetTransformerGraphClassifier,
    DeepSetGraphClassifier,
    GraphSetTransformerClassifier,
    GCNGraphClassifier,
)
from src.graph_set_transformer.data.set_data_set import (
    SetDataset,
    collate_sets,
)

# GraphEncoder imported conditionally in load_drug_drug_from_smiles to avoid deepchem dependency

import matplotlib.pyplot as plt
import pandas as pd


def get_model(model_name, in_channels, hidden_dim, num_classes, dropout=0.1):
    """Get model by name."""
    if model_name == "SetTransformer":
        return SetTransformerGraphClassifier(
            in_channels, hidden_dim, num_classes, dropout=dropout
        )
    elif model_name == "DeepSets":
        return DeepSetGraphClassifier(
            in_channels, hidden_dim, num_classes, dropout=dropout
        )
    elif model_name == "GraphSetConv":
        return GraphSetTransformerClassifier(
            in_channels, hidden_dim, num_classes, dropout=dropout
        )
    elif model_name == "GCN":
        return GCNGraphClassifier(in_channels, hidden_dim, num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_drug_drug_from_graphs(data_dir):
    """
    Load Drug-Drug Interaction dataset from preprocessed graph files.
    Returns train, val, test lists of (drug_pair_graphs, interaction_type).
    """

    def load_split(filename):
        filepath = Path(data_dir) / filename
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        sets = []
        for interaction in data:
            # Combine both drugs as a set (drug1 + drug2)
            drug_pair = interaction["drug1"] + interaction["drug2"]
            interaction_type = int(interaction["interaction_type"])
            # Ensure float features
            for g in drug_pair:
                g.x = g.x.float()
            sets.append((drug_pair, interaction_type))
        return sets

    train_sets = load_split("train_graph_reactions.pkl")
    val_sets = load_split("valid_graph_reactions.pkl")
    test_sets = load_split("test_graph_reactions.pkl")

    return train_sets, val_sets, test_sets


def load_drug_drug_from_smiles(data_dir):
    """
    Load Drug-Drug Interaction dataset from raw SMILES files.
    Uses GraphEncoder to convert SMILES to graphs.
    Returns train, val, test lists of (drug_pair_graphs, label).
    """
    # Import here to avoid deepchem dependency when using preprocessed graphs
    from src.graph_set_transformer.utils.graph_encoder import GraphEncoder

    encoder = GraphEncoder(fix_seed=True)

    def load_split(filename):
        filepath = Path(data_dir) / filename
        with open(filepath, "rb") as f:
            df = pickle.load(f)

        sets = []
        for idx, row in df.iterrows():
            drug1_smiles = row["Drug1"]
            drug2_smiles = row["Drug2"]
            label = int(row["Y"])

            # Convert SMILES to graphs using GraphEncoder
            g1_nx = encoder.smiles_to_nx(drug1_smiles)
            g2_nx = encoder.smiles_to_nx(drug2_smiles)

            if g1_nx is not None and g2_nx is not None:
                g1 = encoder.nx_to_pyg(g1_nx)
                g2 = encoder.nx_to_pyg(g2_nx)

                if g1 is not None and g2 is not None:
                    g1.x = g1.x.float()
                    g2.x = g2.x.float()
                    drug_pair = [g1, g2]
                    sets.append((drug_pair, label))

        return sets

    train_sets = load_split("train.pkl")
    val_sets = load_split("valid.pkl")
    test_sets = load_split("test.pkl")

    return train_sets, val_sets, test_sets


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, set_batch, targets in loader:
        data = data.to(device)
        set_batch = set_batch.to(device)
        targets = targets.to(device).long()

        optimizer.zero_grad()
        pred = model(data, set_batch)
        loss = F.cross_entropy(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, num_classes=None):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for data, set_batch, targets in loader:
            data = data.to(device)
            set_batch = set_batch.to(device)
            logits = model(data, set_batch)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    # Compute AUROC and AUPRC using one-vs-rest strategy for multi-class
    try:
        auroc = roc_auc_score(
            all_targets, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        auroc = float("nan")

    try:
        # For AUPRC, we need to one-hot encode targets
        from sklearn.preprocessing import label_binarize

        if num_classes is None:
            num_classes = all_probs.shape[1]
        targets_onehot = label_binarize(all_targets, classes=list(range(num_classes)))
        auprc = average_precision_score(targets_onehot, all_probs, average="macro")
    except ValueError:
        auprc = float("nan")

    return acc, f1, auroc, auprc


def main():
    data_dir = Path(__file__).parent.parent / "data" / "ddi"
    output_dir = Path(__file__).parent.parent / "results" / "ddi"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Training parameters
    model_names = ["SetTransformer", "DeepSets", "GraphSetConv", "GCN"]
    num_epochs = 200
    hidden_dim = 64
    batch_size = 64
    learning_rate = 1e-3
    dropout = 0.2
    seeds = [10, 20, 30, 40, 50]

    print(f"\n{'#'*70}")
    print("Drug-Drug Interaction Classification")
    print(f"Models: {model_names}")
    print(f"Seeds: {seeds}")
    print(f"{'#'*70}\n")

    print("Loading Drug-Drug Interaction dataset...")
    try:
        train_sets, val_sets, test_sets = load_drug_drug_from_graphs(data_dir)
        print("Loaded from preprocessed graph files")
    except FileNotFoundError:
        print("Graph files not found, converting from SMILES...")
        train_sets, val_sets, test_sets = load_drug_drug_from_smiles(data_dir)
        print("Converted SMILES to graphs using GraphEncoder")

    print(f"Train: {len(train_sets)}, Val: {len(val_sets)}, Test: {len(test_sets)}")

    # Get input dimensions and number of classes
    in_channels = train_sets[0][0][0].x.shape[1]
    all_labels = [s[1] for s in train_sets]
    num_classes = max(all_labels) + 1
    print(f"Input channels: {in_channels}, Num classes: {num_classes}")

    # Storage for aggregated results
    all_seed_results = defaultdict(lambda: defaultdict(list))

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"Seed {seed} ({seed_idx + 1}/{len(seeds)})")
        print(f"{'='*70}")

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Shuffle sets for this seed
        random.shuffle(train_sets)

        # Create DataLoaders
        train_loader = TorchDataLoader(
            SetDataset(train_sets),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_sets,
        )
        val_loader = TorchDataLoader(
            SetDataset(val_sets),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sets,
        )
        test_loader = TorchDataLoader(
            SetDataset(test_sets),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sets,
        )

        all_results = {
            name: {
                "train_loss": [],
                "val_acc": [],
                "val_f1": [],
                "val_auroc": [],
                "val_auprc": [],
                "test_acc": None,
                "test_f1": None,
                "test_auroc": None,
                "test_auprc": None,
                "best_model": None,
            }
            for name in model_names
        }

        # Train each model
        for model_name in model_names:
            print(f"\n{'='*50}")
            print(f"Training {model_name} (Seed {seed})")
            print(f"{'='*50}")

            model = get_model(
                model_name, in_channels, hidden_dim, num_classes, dropout=dropout
            )
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-5
            )

            best_val_acc = 0
            best_val_f1 = 0
            best_val_auroc = 0
            best_val_auprc = 0
            for epoch in range(num_epochs):
                train_loss = train_epoch(model, train_loader, optimizer, device)
                scheduler.step()
                val_acc, val_f1, val_auroc, val_auprc = evaluate(
                    model, val_loader, device, num_classes
                )

                all_results[model_name]["train_loss"].append(train_loss)
                all_results[model_name]["val_acc"].append(val_acc)
                all_results[model_name]["val_f1"].append(val_f1)
                all_results[model_name]["val_auroc"].append(val_auroc)
                all_results[model_name]["val_auprc"].append(val_auprc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_val_auroc = val_auroc
                    best_val_auprc = val_auprc
                    all_results[model_name]["best_model"] = model.state_dict().copy()

                if (epoch + 1) % 10 == 0:
                    print(
                        f"  Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}"
                    )

            # Test evaluation with best model
            model.load_state_dict(all_results[model_name]["best_model"])
            test_acc, test_f1, test_auroc, test_auprc = evaluate(
                model, test_loader, device, num_classes
            )
            all_results[model_name]["test_acc"] = test_acc
            all_results[model_name]["test_f1"] = test_f1
            all_results[model_name]["test_auroc"] = test_auroc
            all_results[model_name]["test_auprc"] = test_auprc

            print(
                f"Best Val Acc: {best_val_acc:.4f}, Val F1: {best_val_f1:.4f}, AUROC: {best_val_auroc:.4f}, AUPRC: {best_val_auprc:.4f}"
            )
            print(
                f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}"
            )

            # Store for aggregation
            all_seed_results[model_name]["val_acc"].append(best_val_acc)
            all_seed_results[model_name]["val_f1"].append(best_val_f1)
            all_seed_results[model_name]["val_auroc"].append(best_val_auroc)
            all_seed_results[model_name]["val_auprc"].append(best_val_auprc)
            all_seed_results[model_name]["test_acc"].append(test_acc)
            all_seed_results[model_name]["test_f1"].append(test_f1)
            all_seed_results[model_name]["test_auroc"].append(test_auroc)
            all_seed_results[model_name]["test_auprc"].append(test_auprc)

        # Save plots for this seed
        seed_output_dir = output_dir / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for model_name in model_names:
            axes[0, 0].plot(all_results[model_name]["train_loss"], label=model_name)
            axes[0, 1].plot(all_results[model_name]["val_acc"], label=model_name)
            axes[0, 2].plot(all_results[model_name]["val_f1"], label=model_name)
            axes[1, 0].plot(all_results[model_name]["val_auroc"], label=model_name)
            axes[1, 1].plot(all_results[model_name]["val_auprc"], label=model_name)

        axes[0, 0].set_title("Train Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()

        axes[0, 1].set_title("Validation Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()

        axes[0, 2].set_title("Validation F1 (Macro)")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("F1")
        axes[0, 2].legend()

        axes[1, 0].set_title("Validation AUROC")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUROC")
        axes[1, 0].legend()

        axes[1, 1].set_title("Validation AUPRC")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("AUPRC")
        axes[1, 1].legend()

        axes[1, 2].axis("off")  # Empty subplot

        plt.tight_layout()
        plt.savefig(seed_output_dir / "training_curves.png", dpi=150)
        plt.close()

        # Save summary for this seed
        summary = pd.DataFrame(
            {
                "Model": model_names,
                "Best Val Acc": [max(all_results[m]["val_acc"]) for m in model_names],
                "Best Val F1": [max(all_results[m]["val_f1"]) for m in model_names],
                "Best Val AUROC": [
                    max(all_results[m]["val_auroc"]) for m in model_names
                ],
                "Best Val AUPRC": [
                    max(all_results[m]["val_auprc"]) for m in model_names
                ],
                "Test Acc": [all_results[m]["test_acc"] for m in model_names],
                "Test F1": [all_results[m]["test_f1"] for m in model_names],
                "Test AUROC": [all_results[m]["test_auroc"] for m in model_names],
                "Test AUPRC": [all_results[m]["test_auprc"] for m in model_names],
            }
        )
        summary.to_csv(seed_output_dir / "results_summary.csv", index=False)
        print(f"\nSaved results to {seed_output_dir}")

    # Aggregate results across all seeds
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}\n")

    aggregated_summary = pd.DataFrame(
        {
            "Model": model_names,
            "Val_Acc_Mean": [
                np.mean(all_seed_results[m]["val_acc"]) for m in model_names
            ],
            "Val_Acc_Std": [
                np.std(all_seed_results[m]["val_acc"]) for m in model_names
            ],
            "Val_F1_Mean": [
                np.mean(all_seed_results[m]["val_f1"]) for m in model_names
            ],
            "Val_F1_Std": [np.std(all_seed_results[m]["val_f1"]) for m in model_names],
            "Val_AUROC_Mean": [
                np.mean(all_seed_results[m]["val_auroc"]) for m in model_names
            ],
            "Val_AUROC_Std": [
                np.std(all_seed_results[m]["val_auroc"]) for m in model_names
            ],
            "Val_AUPRC_Mean": [
                np.mean(all_seed_results[m]["val_auprc"]) for m in model_names
            ],
            "Val_AUPRC_Std": [
                np.std(all_seed_results[m]["val_auprc"]) for m in model_names
            ],
            "Test_Acc_Mean": [
                np.mean(all_seed_results[m]["test_acc"]) for m in model_names
            ],
            "Test_Acc_Std": [
                np.std(all_seed_results[m]["test_acc"]) for m in model_names
            ],
            "Test_F1_Mean": [
                np.mean(all_seed_results[m]["test_f1"]) for m in model_names
            ],
            "Test_F1_Std": [
                np.std(all_seed_results[m]["test_f1"]) for m in model_names
            ],
            "Test_AUROC_Mean": [
                np.mean(all_seed_results[m]["test_auroc"]) for m in model_names
            ],
            "Test_AUROC_Std": [
                np.std(all_seed_results[m]["test_auroc"]) for m in model_names
            ],
            "Test_AUPRC_Mean": [
                np.mean(all_seed_results[m]["test_auprc"]) for m in model_names
            ],
            "Test_AUPRC_Std": [
                np.std(all_seed_results[m]["test_auprc"]) for m in model_names
            ],
        }
    )

    aggregated_summary.to_csv(output_dir / "aggregated_results.csv", index=False)
    print(aggregated_summary)
    print(f"\nSaved to {output_dir / 'aggregated_results.csv'}")

    print(f"\n{'#'*70}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
