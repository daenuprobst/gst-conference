import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch


class SetDataset(Dataset):
    def __init__(self, sets):
        self.sets = sets

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        return self.sets[idx]


class BalancedSetBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_classes=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Group set indices by their label
        self.class_indices = {i: [] for i in range(num_classes)}
        for idx, (graph_set, label) in enumerate(dataset):
            self.class_indices[label].append(idx)

        # Calculate how many sets per class in each batch
        self.sets_per_class = batch_size // num_classes

        # Calculate total number of balanced batches we can create
        min_class_count = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_count // self.sets_per_class

    def __iter__(self):
        # Shuffle indices within each class
        shuffled_indices = {
            class_id: np.random.permutation(indices).tolist()
            for class_id, indices in self.class_indices.items()
        }

        batches = []
        for batch_idx in range(self.num_batches):
            batch = []
            for class_id in range(self.num_classes):
                start = batch_idx * self.sets_per_class
                end = start + self.sets_per_class
                batch.extend(shuffled_indices[class_id][start:end])

            # Shuffle within batch so classes aren't always in same order
            np.random.shuffle(batch)
            batches.append(batch)

        # Shuffle the order of batches
        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.num_batches


def collate_sets(batch_of_sets, verbose=False):
    all_graphs = []
    set_assignments = []
    labels = []
    class_counts = {}

    for set_idx, (graph_set, label) in enumerate(batch_of_sets):
        all_graphs.extend(graph_set)
        set_assignments.extend([set_idx] * len(graph_set))
        labels.append(label)
        class_counts[label] = class_counts.get(label, 0) + 1

    if verbose:
        print(f"Batch class distribution: {class_counts}")

    return (
        Batch.from_data_list(all_graphs),
        torch.tensor(set_assignments, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# def make_label_homogeneous_sets(dataset, set_size):
#     # Group by label
#     label_groups = defaultdict(list)
#     for data in dataset:
#         label_groups[int(data.y.item())].append(data)

#     sets = []

#     for label, graphs in label_groups.items():
#         random.shuffle(graphs)
#         for i in range(0, len(graphs), set_size):
#             sets.append((graphs[i : i + set_size], label))

#     random.shuffle(sets)
#     return sets


def make_label_homogeneous_sets(dataset, set_size, shuffle=False):
    # Group by label
    label_groups = defaultdict(list)
    for data in dataset:
        label_groups[int(data.y.item())].append(data)

    sets = []
    for label, graphs in label_groups.items():
        n = len(graphs)

        # For each graph, create a set with it and (set_size - 1) random others
        used = set()
        for i in range(n):
            # if i in used:
            #     continue
            # Start with the current graph
            current_set = [graphs[i]]

            # Add (set_size - 1) random other graphs from the same label
            # Sample with replacement if we don't have enough graphs
            other_indices = [j for j in range(n) if j != i]

            if len(other_indices) >= set_size - 1:
                # Sample without replacement
                sampled_indices = random.sample(other_indices, set_size - 1)
            else:
                # Sample with replacement if we don't have enough graphs
                sampled_indices = random.choices(other_indices, k=set_size - 1)

            used.update(sampled_indices)

            current_set.extend([graphs[j] for j in sampled_indices])
            sets.append((current_set, label))
    if shuffle:
        random.shuffle(sets)

    return sets


def make_label_homogeneous_sets_rand_card(dataset, min_size=1, max_size=10):
    label_groups = defaultdict(list)
    for data in dataset:
        label_groups[int(data.y.item())].append(data)

    sets = []
    for label, graphs in label_groups.items():
        random.shuffle(graphs)
        i = 0
        while i < len(graphs):
            remaining = len(graphs) - i
            current_set_size = random.randint(min_size, min(max_size, remaining))

            sets.append((graphs[i : i + current_set_size], label))
            i += current_set_size

    random.shuffle(sets)
    return sets
