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

from graph_set_transformer.data import (
    SetDataset,
    collate_sets,
    make_label_homogeneous_sets,
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
    dataset_names = ["BACE"]
    model_names = ["SetTransformer", "DeepSets", "GraphSetConv"]
    num_epochs = 100
    hidden_dim = 64
    batch_size = 32
    set_size = 10
    learning_rate = 1e-3

    all_results = {
        "SetTransformer": {"train_loss": [], "val_auroc": []},
        "DeepSets": {"train_loss": [], "val_auroc": []},
        "GraphSetConv": {"train_loss": [], "val_auroc": []},
    }

    # Load dataset (using first dataset for now)
    dataset = load_dataset(dataset_names[0])

    # Get input dimensions from dataset
    in_channels = dataset[0].x.shape[1]
    num_classes = 2  # Binary classification for e.g. bace

    # Split dataset
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    indices = list(range(n))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    # Create sets of graphs with homogeneous labels
    train_sets = make_label_homogeneous_sets(train_dataset, set_size)
    val_sets = make_label_homogeneous_sets(val_dataset, set_size)
    test_sets = make_label_homogeneous_sets(test_dataset, set_size)

    # Create SetDatasets
    train_set_dataset = SetDataset(train_sets)
    val_set_dataset = SetDataset(val_sets)
    test_set_dataset = SetDataset(test_sets)

    # Create DataLoaders
    train_loader = TorchDataLoader(
        train_set_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_sets
    )
    val_loader = TorchDataLoader(
        val_set_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sets
    )
    test_loader = TorchDataLoader(
        test_set_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sets
    )

    # Train each model
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")

        model = get_model(model_name, in_channels, hidden_dim, num_classes)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_auroc = 0
        for epoch in range(num_epochs):
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device)

            # Validate
            val_auroc = evaluate(model, val_loader, device)

            # Save Results
            all_results[model_name]["train_loss"].append(train_loss)
            all_results[model_name]["val_auroc"].append(val_auroc)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val AUROC: {val_auroc:.4f}"
                )

        print(f"Best Val AUROC for {model_name}: {best_val_auroc:.4f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name in model_names:
        axes[0].plot(all_results[model_name]["train_loss"], label=model_name)
        axes[1].plot(all_results[model_name]["val_auroc"], label=model_name)

    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print("\nSaved plot to model_comparison.png")

    # Summary Table
    summary = pd.DataFrame(
        {
            "Model": model_names,
            "Best Val AUROC": [max(all_results[m]["val_auroc"]) for m in model_names],
            "Final Train Loss": [all_results[m]["train_loss"][-1] for m in model_names],
        }
    )
    summary.to_csv("model_comparison.csv", index=False)
    print("\nSaved summary to model_comparison.csv")
    print(summary)


if __name__ == "__main__":
    main()
