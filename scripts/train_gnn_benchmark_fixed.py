"""Training script for GNNBenchmarkDataset (MNIST, CIFAR10) with fixed set sizes."""

import torch
import torch.nn.functional as F
import numpy as np
import random
import sys
from collections import defaultdict
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import GNNBenchmarkDataset
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from pathlib import Path

from graph_set_transformer.models.model_dropout import (
    SetTransformerGraphClassifier,
    DeepSetGraphClassifier,
    SetGraphClassifier,
    SetDataset,
    collate_sets,
    make_label_homogeneous_sets,
)

import matplotlib.pyplot as plt
import pandas as pd


def get_model(model_name, in_channels, hidden_dim, num_classes, dropout=0.3):
    if model_name == "SetTransformer":
        return SetTransformerGraphClassifier(
            in_channels, hidden_dim, num_classes, dropout=dropout
        )
    elif model_name == "DeepSets":
        return DeepSetGraphClassifier(
            in_channels, hidden_dim, num_classes, dropout=dropout
        )
    elif model_name == "GraphSetConv":
        return SetGraphClassifier(in_channels, hidden_dim, num_classes, dropout=dropout)


def load_gnn_benchmark_dataset(dataset_name):
    """Load GNNBenchmarkDataset with official splits"""
    train_dataset = GNNBenchmarkDataset(root="./data", name=dataset_name, split="train")
    val_dataset = GNNBenchmarkDataset(root="./data", name=dataset_name, split="val")
    test_dataset = GNNBenchmarkDataset(root="./data", name=dataset_name, split="test")

    # Ensure float features
    for dataset in [train_dataset, val_dataset, test_dataset]:
        for data in dataset:
            data.x = data.x.float()

    return train_dataset, val_dataset, test_dataset


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, set_batch, targets in loader:
        data = data.to(device)
        set_batch = set_batch.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        pred = model(data, set_batch)
        loss = F.cross_entropy(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, set_batch, targets in loader:
            data = data.to(device)
            set_batch = set_batch.to(device)
            logits = model(data, set_batch)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.hstack(all_labels)

    if num_classes == 2:
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
        precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
        auprc = auc(recall, precision)
    else:
        # Multi-class: macro-averaged metrics
        auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        auprc = average_precision_score(all_labels, all_probs, average="macro")

    return auroc, auprc


def main():

    dataset_name = sys.argv[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    model_names = ["SetTransformer", "DeepSets", "GraphSetConv"]
    num_epochs = 200
    hidden_dim = 64
    batch_size = 16
    learning_rate = 1e-3
    dropout = 0.1

    # Grid search parameters
    set_sizes = [10]
    seeds = [10, 20, 30, 40, 50]

    print(f"\n{'#'*70}")
    print(f"{dataset_name} Grid Search Training (Fixed Set Size)")
    print(f"Official GNNBenchmarkDataset Splits")
    print(f"Set Sizes: {set_sizes}")
    print(f"Seeds: {seeds}")
    print(
        f"Total experiments: {len(set_sizes)} × {len(seeds)} = {len(set_sizes) * len(seeds)}"
    )
    print(f"{'#'*70}\n")

    # Load official splits
    print("Loading GNNBenchmarkDataset...")
    train_dataset, val_dataset, test_dataset = load_gnn_benchmark_dataset(dataset_name)
    print(
        f"Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    # Get dataset properties
    in_channels = train_dataset[0].x.shape[1]
    num_classes = train_dataset.num_classes
    print(f"Input channels: {in_channels}, Num classes: {num_classes}\n")

    total_experiments = len(set_sizes) * len(seeds)
    experiment_count = 0

    for set_size in set_sizes:
        print(f"\n{'*'*70}")
        print(f"SET SIZE: {set_size}")
        print(f"{'*'*70}")

        # Base output directory
        base_output_dir = (
            Path("../results") / dataset_name / f"{dataset_name}_set_{set_size}"
        )
        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for aggregated results
        all_seed_results = defaultdict(lambda: defaultdict(list))

        for seed in seeds:
            experiment_count += 1
            print(f"\n{'='*70}")
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Set Size: {set_size}, Seed: {seed}")
            print(f"{'='*70}\n")

            # Set seeds
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

            # Output directory
            output_dir = base_output_dir / f"seed_{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)

            all_results = {
                "SetTransformer": {
                    "train_loss": [],
                    "val_auroc": [],
                    "val_auprc": [],
                    "test_auroc": None,
                    "test_auprc": None,
                    "best_model": None,
                },
                "DeepSets": {
                    "train_loss": [],
                    "val_auroc": [],
                    "val_auprc": [],
                    "test_auroc": None,
                    "test_auprc": None,
                    "best_model": None,
                },
                "GraphSetConv": {
                    "train_loss": [],
                    "val_auroc": [],
                    "val_auprc": [],
                    "test_auroc": None,
                    "test_auprc": None,
                    "best_model": None,
                },
            }

            # Create sets (fixed size for train/val/test)
            print(f"Creating label-homogeneous sets (size={set_size})...")
            train_sets = make_label_homogeneous_sets(list(train_dataset), set_size)
            val_sets = make_label_homogeneous_sets(list(val_dataset), set_size)
            test_sets = make_label_homogeneous_sets(list(test_dataset), set_size)
            print(
                f"Created sets: Train={len(train_sets)}, Val={len(val_sets)}, Test={len(test_sets)}"
            )

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

                best_val_auroc = 0
                best_val_auprc = 0
                for epoch in range(num_epochs):
                    train_loss = train_epoch(model, train_loader, optimizer, device)
                    val_auroc, val_auprc = evaluate(
                        model, val_loader, device, num_classes
                    )

                    all_results[model_name]["train_loss"].append(train_loss)
                    all_results[model_name]["val_auroc"].append(val_auroc)
                    all_results[model_name]["val_auprc"].append(val_auprc)

                    if val_auroc > best_val_auroc:
                        best_val_auroc = val_auroc
                        best_val_auprc = val_auprc
                        all_results[model_name][
                            "best_model"
                        ] = model.state_dict().copy()

                    if (epoch + 1) % 20 == 0:
                        print(
                            f"  Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}"
                        )

                print(
                    f"Best Val AUROC: {best_val_auroc:.4f}, AUPRC: {best_val_auprc:.4f}"
                )

                # Evaluate on test set
                model.load_state_dict(all_results[model_name]["best_model"])
                test_auroc, test_auprc = evaluate(
                    model, test_loader, device, num_classes
                )
                all_results[model_name]["test_auroc"] = test_auroc
                all_results[model_name]["test_auprc"] = test_auprc
                print(f"Test AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}")

                # Store for aggregation
                all_seed_results[model_name]["val_auroc"].append(best_val_auroc)
                all_seed_results[model_name]["val_auprc"].append(best_val_auprc)
                all_seed_results[model_name]["test_auroc"].append(test_auroc)
                all_seed_results[model_name]["test_auprc"].append(test_auprc)

            # Save plots and results for this seed
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            for model_name in model_names:
                axes[0, 0].plot(all_results[model_name]["train_loss"], label=model_name)
                axes[0, 1].plot(all_results[model_name]["val_auroc"], label=model_name)
                axes[0, 2].plot(all_results[model_name]["val_auprc"], label=model_name)

            axes[0, 0].set_title("Train Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()

            axes[0, 1].set_title("Validation AUROC")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("AUROC")
            axes[0, 1].legend()

            axes[0, 2].set_title("Validation AUPRC")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("AUPRC")
            axes[0, 2].legend()

            # Test metrics
            test_aurocs = [all_results[m]["test_auroc"] for m in model_names]
            test_auprcs = [all_results[m]["test_auprc"] for m in model_names]

            axes[1, 0].axis("off")

            axes[1, 1].bar(
                model_names, test_aurocs, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
            axes[1, 1].set_title(f"Test AUROC (Seed {seed})")
            axes[1, 1].set_ylabel("AUROC")
            axes[1, 1].set_ylim([0, 1])
            for i, (name, auroc) in enumerate(zip(model_names, test_aurocs)):
                axes[1, 1].text(
                    i, auroc + 0.02, f"{auroc:.4f}", ha="center", va="bottom"
                )

            axes[1, 2].bar(
                model_names, test_auprcs, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
            axes[1, 2].set_title(f"Test AUPRC (Seed {seed})")
            axes[1, 2].set_ylabel("AUPRC")
            axes[1, 2].set_ylim([0, 1])
            for i, (name, auprc) in enumerate(zip(model_names, test_auprcs)):
                axes[1, 2].text(
                    i, auprc + 0.02, f"{auprc:.4f}", ha="center", va="bottom"
                )

            plt.tight_layout()
            plt.savefig(output_dir / "training_curves.png", dpi=150)
            plt.close()

            # Save CSV results
            summary = pd.DataFrame(
                {
                    "Model": model_names,
                    "Best Val AUROC": [
                        max(all_results[m]["val_auroc"]) for m in model_names
                    ],
                    "Best Val AUPRC": [
                        max(all_results[m]["val_auprc"]) for m in model_names
                    ],
                    "Test AUROC": [all_results[m]["test_auroc"] for m in model_names],
                    "Test AUPRC": [all_results[m]["test_auprc"] for m in model_names],
                }
            )
            summary.to_csv(output_dir / "results_summary.csv", index=False)
            print(f"\nSaved results to {output_dir}")

        # Aggregate results for this set size
        print(f"\n{'='*70}")
        print(f"AGGREGATED RESULTS FOR SET SIZE {set_size}")
        print(f"{'='*70}\n")

        aggregated_summary = pd.DataFrame(
            {
                "Model": model_names,
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

        aggregated_summary.to_csv(
            base_output_dir / "aggregated_results.csv", index=False
        )
        print(aggregated_summary)
        print(f"\nSaved to {base_output_dir / 'aggregated_results.csv'}")

    print(f"\n{'#'*70}")
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: ../results/{dataset_name}/")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
