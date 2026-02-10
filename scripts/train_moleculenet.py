import torch
import torch_geometric
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import roc_auc_score
from pathlib import Path
from datetime import datetime
from graph_set_transformer.models import (
    SetTransformerGraphClassifier,
    DeepSetGraphClassifier,
    GraphSetTransformerGraphClassifier,
)

from graph_set_transformer.utils import molecule_net_loader

from graph_set_transformer.data import (
    SetDataset,
    BalancedSetBatchSampler,
    collate_sets,
    make_label_homogeneous_sets,
    make_label_homogeneous_sets_rand_card,
)

import matplotlib.pyplot as plt
import pandas as pd


def get_model(model_name, in_channels, hidden_dim, num_classes):
    if model_name == "SetTransformer":
        return SetTransformerGraphClassifier(in_channels, hidden_dim, num_classes)
    elif model_name == "DeepSets":
        return DeepSetGraphClassifier(in_channels, hidden_dim, num_classes)
    elif model_name == "GraphSetConv":
        return GraphSetTransformerGraphClassifier(in_channels, hidden_dim, num_classes)


def load_dataset(dataset_name):
    """Load datasets"""

    def transform(data):
        data.x = data.x.float()
        return data

    dataset = MoleculeNet(root="./data", name=dataset_name, pre_transform=transform)
    return dataset


def calculate_class_weights(dataset, num_classes=2):
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for data in dataset:
        class_counts[data.y.item()] += 1

    # Calculate weights as inverse frequency
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)

    # Normalize weights (optional, but helps with numerical stability)
    class_weights = class_weights / class_weights.sum() * num_classes

    print(f"\nClass distribution in training set:")
    for i in range(num_classes):
        print(
            f"  Class {i}: {int(class_counts[i])} samples ({class_counts[i]/total_samples*100:.2f}%)"
        )
    print(f"Class weights: {class_weights.tolist()}")

    return class_weights


def train_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    for data, set_batch, targets in loader:
        data = data.to(device)
        set_batch = set_batch.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        pred = model(data, set_batch)

        # Use weighted cross entropy if class weights are provided
        if class_weights is not None:
            loss = F.cross_entropy(pred, targets, weight=class_weights)
        else:
            loss = F.cross_entropy(pred, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for data, set_batch, targets in loader:
            data = data.to(device)
            set_batch = set_batch.to(device)
            logits = model(data, set_batch)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return roc_auc_score(all_targets, all_probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    model_names = ["SetTransformer", "DeepSets", "GraphSetConv"]
    # model_names = ["GraphSetConv"]
    # set_sizes = [5, 10, 20]
    set_sizes = [10]
    num_epochs = 500
    hidden_dim = 64
    batch_size = 32
    learning_rate = 1e-4
    num_trials = 5
    use_class_weights = False

    all_results = {
        set_size: {
            model_name: {
                "train_loss_per_trial": [],
                "val_auroc_per_trial": [],
                "test_auroc_per_trial": [],
            }
            for model_name in model_names
        }
        for set_size in set_sizes
    }

    train_dataset, val_dataset, test_dataset, tasks = molecule_net_loader(
        "bace", "data/moleculenet/bace.csv.xz"
    )

    in_channels = train_dataset[0].x.shape[1]
    num_classes = 2

    # Calculate class weights from the training dataset
    if use_class_weights:
        class_weights = calculate_class_weights(train_dataset, num_classes)
        class_weights = class_weights.to(device)
        print(f"\nUsing class weights: {class_weights.cpu().tolist()}\n")
    else:
        class_weights = None
        print("\nNot using class weights\n")

    # Iterate over set sizes
    for set_size in set_sizes:
        print(f"\n{'*'*70}")
        print(f"* SET SIZE: {set_size}")
        print(f"{'*'*70}")

        # Create sets of graphs with homogeneous labels
        val_sets = make_label_homogeneous_sets(val_dataset, 1)
        test_sets = make_label_homogeneous_sets(test_dataset, 1)

        # Create SetDatasets
        val_set_dataset = SetDataset(val_sets)
        test_set_dataset = SetDataset(test_sets)

        # Create DataLoaders
        val_loader = TorchDataLoader(
            val_set_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sets,
        )
        test_loader = TorchDataLoader(
            test_set_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sets,
        )

        for trial in range(num_trials):
            print(f"\n{'#'*60}")
            print(f"# SET SIZE: {set_size} - TRIAL {trial + 1}/{num_trials}")
            print(f"{'#'*60}")

            # Results for this trial (for plotting the first trial)
            trial_results = {
                model_name: {"train_loss": [], "val_auroc": []}
                for model_name in model_names
            }

            # Train each model
            for model_name in model_names:
                print(f"\n{'='*50}")
                print(
                    f"Set Size {set_size} - Trial {trial + 1} - Training {model_name}"
                )
                print(f"{'='*50}")

                model = get_model(model_name, in_channels, hidden_dim, num_classes)
                model = model.to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                )

                best_val_auroc = 0
                best_model_state = None

                for epoch in range(num_epochs):
                    # train_sets = make_label_homogeneous_sets_rand_card(
                    #     train_dataset, max_size=10
                    # )
                    train_sets = make_label_homogeneous_sets(
                        train_dataset, set_size, shuffle=True
                    )
                    train_set_dataset = SetDataset(train_sets)

                    balanced_sampler = BalancedSetBatchSampler(
                        train_set_dataset, batch_size=batch_size, num_classes=2
                    )

                    train_loader = TorchDataLoader(
                        train_set_dataset,
                        # batch_size=batch_size,
                        # shuffle=True,
                        batch_sampler=balanced_sampler,
                        collate_fn=collate_sets,
                    )

                    if epoch == 0 and trial == 0:
                        print(f"\nVerifying batch balance for epoch 1:")
                        for i, (data, set_batch, targets) in enumerate(train_loader):
                            class_counts = torch.bincount(targets)
                            print(f"  Batch {i}: {class_counts.tolist()}")
                            if i >= 2:  # Just show first 3 batches
                                break

                    # Train with class weights
                    train_loss = train_epoch(
                        model, train_loader, optimizer, device, class_weights
                    )

                    # Validate
                    val_auroc = evaluate(model, val_loader, device)

                    # Save Results
                    trial_results[model_name]["train_loss"].append(train_loss)
                    trial_results[model_name]["val_auroc"].append(val_auroc)

                    if val_auroc > best_val_auroc:
                        best_val_auroc = val_auroc
                        best_model_state = {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        }

                    if (epoch + 1) % 10 == 0:
                        print(
                            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val AUROC: {val_auroc:.4f}"
                        )

                print(f"Best Val AUROC for {model_name}: {best_val_auroc:.4f}")

                # Evaluate on test set
                model.load_state_dict(best_model_state)
                test_auroc = evaluate(model, test_loader, device)
                print(f"Test AUROC for {model_name}: {test_auroc:.4f}")

                print(
                    f"Best Val AUROC for {model_name} (Set Size {set_size}, Trial {trial + 1}): {best_val_auroc:.4f}"
                )
                print(
                    f"Test AUROC for {model_name} (Set Size {set_size}, Trial {trial + 1}): {test_auroc:.4f}"
                )

                # Store results across trials
                all_results[set_size][model_name]["train_loss_per_trial"].append(
                    trial_results[model_name]["train_loss"]
                )
                all_results[set_size][model_name]["val_auroc_per_trial"].append(
                    trial_results[model_name]["val_auroc"]
                )
                all_results[set_size][model_name]["test_auroc_per_trial"].append(
                    test_auroc
                )

    print(f"\n{'='*70}")
    print("FINAL RESULTS ACROSS ALL SET SIZES, MODELS, AND TRIALS")
    print(f"{'='*70}\n")

    summary_data = []
    for set_size in set_sizes:
        print(f"\nSET SIZE: {set_size}")
        print(f"{'-'*60}")
        for model_name in model_names:
            test_aurocs = all_results[set_size][model_name]["test_auroc_per_trial"]
            mean_test_auroc = np.mean(test_aurocs)
            std_test_auroc = np.std(test_aurocs)

            print(f"{model_name}:")
            print(f"  Test AUROC across {num_trials} trials: {test_aurocs}")
            print(f"  Mean ± Std: {mean_test_auroc:.4f} ± {std_test_auroc:.4f}\n")

            summary_data.append(
                {
                    "Set Size": set_size,
                    "Model": model_name,
                    "Mean Test AUROC": mean_test_auroc,
                    "Std Test AUROC": std_test_auroc,
                    "All Test AUROCs": test_aurocs,
                }
            )

    # Create plots for each set size using the first trial's data
    num_set_sizes = len(set_sizes)
    fig, axes = plt.subplots(num_set_sizes, 2, figsize=(12, 5 * num_set_sizes))

    # Handle case where there's only one set size
    if num_set_sizes == 1:
        axes = axes.reshape(1, -1)

    for idx, set_size in enumerate(set_sizes):
        for model_name in model_names:
            # Use first trial for plotting
            axes[idx, 0].plot(
                all_results[set_size][model_name]["train_loss_per_trial"][0],
                label=model_name,
            )
            axes[idx, 1].plot(
                all_results[set_size][model_name]["val_auroc_per_trial"][0],
                label=model_name,
            )

        axes[idx, 0].set_title(f"Train Loss (Set Size {set_size}, Trial 1)")
        axes[idx, 0].set_xlabel("Epoch")
        axes[idx, 0].set_ylabel("Loss")
        axes[idx, 0].legend()

        axes[idx, 1].set_title(f"Validation AUROC (Set Size {set_size}, Trial 1)")
        axes[idx, 1].set_xlabel("Epoch")
        axes[idx, 1].set_ylabel("AUROC")
        axes[idx, 1].legend()

    plt.tight_layout()

    # Add class weights and set sizes info to filename if used
    plot_filename = (
        "model_comparison_set_sizes_weighted.png"
        if use_class_weights
        else "model_comparison_set_sizes.png"
    )
    plt.savefig(plot_filename)
    print(f"\nSaved plot to {plot_filename}")

    # Summary Table
    csv_filename = (
        "model_comparison_set_sizes_weighted.csv"
        if use_class_weights
        else "model_comparison_set_sizes.csv"
    )
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_filename, index=False)
    print(f"Saved summary to {csv_filename}")
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary_df[["Set Size", "Model", "Mean Test AUROC", "Std Test AUROC"]])
    print("\n")


if __name__ == "__main__":
    main()
