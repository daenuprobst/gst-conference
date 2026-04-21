import torch
import torch.nn.functional as F
import numpy as np
import random
from time import perf_counter
from datetime import datetime
from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import LRGBDataset
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import average_precision_score

# Assuming these are available in your local environment
from graph_set_transformer.models import (
    SetTransformerGraphClassifier,
    DeepSetGraphClassifier,
    GraphSetTransformerClassifier,
)
from graph_set_transformer.data import SetDataset

# --- Models ---


class GCNBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super(GCNBaseline, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data, set_batch):
        # 1. Node embeddings
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # 2. Graph-level readout (pool nodes into subgraphs)
        # data.batch maps nodes to subgraphs
        x = global_mean_pool(x, data.batch)

        # 3. Set-level readout (pool subgraphs into the peptide set)
        # set_batch maps subgraphs to the set index
        set_x = global_mean_pool(x, set_batch)

        return self.classifier(set_x)


# --- Utility Functions ---


def collate_sets_lrgb(batch):
    """
    Handles multi-label vectors and batched subgraph sets for LRGB.
    """
    all_graphs = []
    set_indices = []
    labels = []

    for set_idx, (graphs, label) in enumerate(batch):
        all_graphs.extend(graphs)
        set_indices.extend([set_idx] * len(graphs))
        labels.append(label)

    batched_data = Batch.from_data_list(all_graphs)
    set_batch = torch.tensor(set_indices, dtype=torch.long)

    if torch.is_tensor(labels[0]):
        batched_labels = torch.stack(labels)
    else:
        batched_labels = torch.tensor(np.array(labels), dtype=torch.float)

    return batched_data, set_batch, batched_labels


def auprc_score(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average="macro")


def get_model(model_name, in_channels, hidden_dim, num_classes):
    if model_name == "SetTransformer":
        return SetTransformerGraphClassifier(in_channels, hidden_dim, num_classes)
    elif model_name == "DeepSets":
        return DeepSetGraphClassifier(in_channels, hidden_dim, num_classes)
    elif model_name == "GraphSetConv":
        return GraphSetTransformerClassifier(in_channels, hidden_dim, num_classes)
    elif model_name == "GCN":
        return GCNBaseline(in_channels, hidden_dim, num_classes)


def split_peptide_into_set(data, set_size=10, hops=2):
    subgraphs = []
    num_nodes = data.num_nodes

    if num_nodes >= set_size:
        indices = random.sample(range(num_nodes), set_size)
    else:
        indices = random.choices(range(num_nodes), k=set_size)

    for idx in indices:
        node_idx, edge_index, edge_mask, _ = k_hop_subgraph(
            node_idx=idx,
            num_hops=hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )

        sub_data = data.clone()
        sub_data.x = data.x[node_idx].float()
        sub_data.edge_index = edge_index
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            sub_data.edge_attr = data.edge_attr[edge_mask]
        subgraphs.append(sub_data)

    return subgraphs, data.y


def prepare_lrgb_set_dataset(dataset, set_size):
    set_list = []
    print(f"Generating sets (size {set_size}) from {len(dataset)} peptides...")
    for data in dataset:
        subgraphs, label = split_peptide_into_set(data, set_size=set_size)
        set_list.append((subgraphs, label))
    return SetDataset(set_list)


# --- Training and Evaluation ---


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, set_batch, targets in loader:
        data, set_batch = data.to(device), set_batch.to(device)
        targets = targets.to(device).float().squeeze()

        optimizer.zero_grad()
        logits = model(data, set_batch)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        loss = F.binary_cross_entropy_with_logits(logits, targets)
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
            data, set_batch = data.to(device), set_batch.to(device)
            targets = targets.float().squeeze()

            logits = model(data, set_batch)
            probs = torch.sigmoid(logits)

            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    return auprc_score(np.concatenate(all_targets), np.concatenate(all_probs))


# --- Main Script ---


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "Peptides-func"
    model_names = ["GCN", "GraphSetConv"]

    learning_rates = {
        "SetTransformer": 1e-3,
        "DeepSets": 1e-3,
        "GraphSetConv": 1e-4,
        "GCN": 1e-3,
    }
    hidden_dims = {"SetTransformer": 64, "DeepSets": 64, "GraphSetConv": 64, "GCN": 64}

    set_sizes = [50]  # Example size
    num_epochs = 50  # Reduced for testing
    batch_size = 32
    num_trials = 1

    # Load and Subset LRGB
    print(f"Loading {dataset_name}...")
    train_raw = LRGBDataset(root="./data/LRGB", name=dataset_name, split="train")
    val_raw = LRGBDataset(root="./data/LRGB", name=dataset_name, split="val")
    test_raw = LRGBDataset(root="./data/LRGB", name=dataset_name, split="test")

    train_raw = train_raw[: len(train_raw) // 10]
    val_raw = val_raw[: len(val_raw) // 10]
    test_raw = test_raw[: len(test_raw) // 10]

    in_channels = train_raw.num_features
    num_classes = train_raw.num_classes

    results_table = []

    for set_size in set_sizes:
        val_set_dataset = prepare_lrgb_set_dataset(val_raw, set_size)
        test_set_dataset = prepare_lrgb_set_dataset(test_raw, set_size)

        val_loader = TorchDataLoader(
            val_set_dataset, batch_size=batch_size, collate_fn=collate_sets_lrgb
        )
        test_loader = TorchDataLoader(
            test_set_dataset, batch_size=batch_size, collate_fn=collate_sets_lrgb
        )

        for trial in range(num_trials):
            train_set_dataset = prepare_lrgb_set_dataset(train_raw, set_size)
            train_loader = TorchDataLoader(
                train_set_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_sets_lrgb,
            )

            for model_name in model_names:
                print(
                    f"\n>> Model: {model_name} | Set Size: {set_size} | Trial: {trial + 1}"
                )
                model = get_model(
                    model_name, in_channels, hidden_dims[model_name], num_classes
                ).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=learning_rates[model_name], weight_decay=0.01
                )

                best_val_ap = 0
                best_model_state = None

                for epoch in range(num_epochs):
                    t_loss = train_epoch(model, train_loader, optimizer, device)
                    v_ap = evaluate(model, val_loader, device)

                    if v_ap > best_val_ap:
                        best_val_ap = v_ap
                        best_model_state = {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        }

                    if (epoch + 1) % 5 == 0:
                        print(
                            f"Epoch {epoch + 1:03d} | Loss: {t_loss:.4f} | Val AP: {v_ap:.4f}"
                        )

                model.load_state_dict(best_model_state)
                test_ap = evaluate(model, test_loader, device)
                print(f"Final Test AP: {test_ap:.4f}")

                results_table.append(
                    {"Model": model_name, "Set Size": set_size, "Test AP": test_ap}
                )

    print("\n" + "=" * 30 + "\nSUMMARY\n" + "=" * 30)
    print(pd.DataFrame(results_table))


if __name__ == "__main__":
    main()
