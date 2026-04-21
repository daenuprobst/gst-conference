import torch
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd

from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.datasets import LRGBDataset
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import average_precision_score

# --- Models ---


class StandardGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data, *args):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GCNSetBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data, set_batch):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        # Pool nodes into subgraph embeddings
        subgraph_x = global_mean_pool(x, data.batch)
        # Pool subgraph embeddings into set embedding
        set_x = global_mean_pool(subgraph_x, set_batch)
        return self.classifier(set_x)


# --- Utility Functions ---


def collate_sets_lrgb(batch):
    """
    batch is a list of tuples: [(list_of_subgraphs, label), ...]
    We return a single Batch object.
    """
    all_subgraphs = []
    set_indices = []
    labels = []

    for set_idx, (subgraphs, label) in enumerate(batch):
        all_subgraphs.extend(subgraphs)
        set_indices.extend([set_idx] * len(subgraphs))
        labels.append(label)

    # Standard PyG Batching
    new_batch = Batch.from_data_list(all_subgraphs)
    new_batch.set_batch = torch.tensor(set_indices, dtype=torch.long)

    if torch.is_tensor(labels[0]):
        new_batch.y = torch.stack(labels)
    else:
        new_batch.y = torch.tensor(np.array(labels), dtype=torch.float)

    return new_batch


def auprc_score(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average="macro")


def get_model(model_name, in_channels, hidden_dim, num_classes):
    if model_name == "GCN_Whole":
        return StandardGCN(in_channels, hidden_dim, num_classes)
    elif model_name == "GCN_Set":
        return GCNSetBaseline(in_channels, hidden_dim, num_classes)
    # Placeholder: assume other models are imported correctly
    return GCNSetBaseline(in_channels, hidden_dim, num_classes)


def split_peptide_into_set(data, set_size=5, hops=2):
    subgraphs = []
    num_nodes = data.num_nodes
    indices = random.sample(range(num_nodes), min(set_size, num_nodes))
    if len(indices) < set_size:
        indices += random.choices(range(num_nodes), k=set_size - len(indices))

    for idx in indices:
        node_idx, edge_index, edge_mask, _ = k_hop_subgraph(
            idx, hops, data.edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        sub_data = Data(x=data.x[node_idx], edge_index=edge_index)
        subgraphs.append(sub_data)
    return subgraphs, data.y


# --- Core Loop ---


def train_epoch(model, loader, optimizer, device, is_set_model):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        targets = data.y.float().squeeze()

        optimizer.zero_grad()
        if is_set_model:
            logits = model(data, data.set_batch)
        else:
            logits = model(data)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, is_set_model):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            targets = data.y.float().squeeze()

            if is_set_model:
                logits = model(data, data.set_batch)
            else:
                logits = model(data)

            probs = torch.sigmoid(logits)
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return auprc_score(np.concatenate(all_targets), np.concatenate(all_probs))


# --- Main ---


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "Peptides-func"

    # Load 1/10th subset
    full_train = LRGBDataset(root="./data/LRGB", name=dataset_name, split="train")[
        :1500
    ]
    full_val = LRGBDataset(root="./data/LRGB", name=dataset_name, split="val")[:200]
    full_test = LRGBDataset(root="./data/LRGB", name=dataset_name, split="test")[:200]

    in_channels, num_classes = full_train.num_features, full_train.num_classes
    set_size = 1

    # 1. Whole Graph Loaders
    whole_val_loader = PyGDataLoader(full_val, batch_size=32)
    whole_test_loader = PyGDataLoader(full_test, batch_size=32)
    whole_train_loader = PyGDataLoader(full_train, batch_size=32, shuffle=True)

    # 2. Split Set Loaders
    val_set_data = [split_peptide_into_set(d, set_size) for d in full_val]
    test_set_data = [split_peptide_into_set(d, set_size) for d in full_test]

    val_set_loader = PyGDataLoader(
        val_set_data, batch_size=32, collate_fn=collate_sets_lrgb
    )
    test_set_loader = PyGDataLoader(
        test_set_data, batch_size=32, collate_fn=collate_sets_lrgb
    )

    model_names = ["GCN_Whole", "GCN_Set"]
    results = []

    for name in model_names:
        print(f"\n--- Testing {name} ---")
        is_set = name == "GCN_Set"
        model = get_model(name, in_channels, 64, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        v_loader = val_set_loader if is_set else whole_val_loader
        t_loader = test_set_loader if is_set else whole_test_loader

        best_ap = 0
        for epoch in range(200):
            # For Training: re-split to get fresh subgraph samples
            if is_set:
                train_set_data = [
                    split_peptide_into_set(d, set_size) for d in full_train
                ]
                curr_train_loader = PyGDataLoader(
                    train_set_data,
                    batch_size=32,
                    shuffle=True,
                    collate_fn=collate_sets_lrgb,
                )
            else:
                curr_train_loader = whole_train_loader

            loss = train_epoch(model, curr_train_loader, optimizer, device, is_set)
            v_ap = evaluate(model, v_loader, device, is_set)

            if v_ap > best_ap:
                best_ap = v_ap
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1:02d} | Loss: {loss:.4f} | Val AP: {v_ap:.4f}")

        results.append({"Model": name, "Best Val AP": best_ap})

    print("\n", pd.DataFrame(results))


if __name__ == "__main__":
    main()
