"""
oversquash_data.py
==================

Synthetic dataset for testing whether interleaved set-graph architectures
(GraphSetConv) handle graph over-squashing better than standard GNNs +
pipeline set heads.

Construction
------------
Each example is a "dumbbell" graph:
- K clusters of N nodes each, each cluster densely connected internally
  (Erdős–Rényi with high p)
- Adjacent clusters connected by exactly B edges (the "bottleneck")
- For K=2 (the headline): one bottleneck between the two clusters
- For K>2 (the chain ablation): K-1 bottlenecks, one between each pair

Each node carries a categorical label in {0..L-1}, sampled uniformly.

Target
------
Count of cross-cluster pairs (i, j) with i in cluster X, j in cluster Y,
where label_i == label_j, normalized by N^2 so the target is in [0, 1].
- For K=2 (dumbbell): X=0, Y=1
- For K>2 (chain): X=0, Y=K-1 (the *endpoints* of the chain)
  This forces long-range information to traverse all K-1 bottlenecks.

This task forces cross-cluster reasoning. A standard GNN with depth d can
only see d-hop neighborhoods; information from cluster Y to cluster X has
to flow through every bottleneck, with severe averaging at each.
GSC with the cut (clusters as a set of disconnected graphs) lets the
set-level transformer mix information directly.

Two views of every example
--------------------------
- "uncut": single graph with all clusters and all bottlenecks present
- "cut": K disconnected graphs (the bottleneck edges are dropped)

The benchmark runs each model on its appropriate view:
- GCN-uncut: receives the uncut graph; degrades as B shrinks
- GCN+set-head, GSC variants: receive the cut graphs
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch


# Node features = one-hot label, dim = NUM_LABELS
NUM_LABELS = 4


@dataclass
class DumbbellSample:
    """One synthetic example. Both views are pre-computed.

    The 'uncut' view has all bottleneck edges. The 'cut' view drops them,
    leaving K disconnected components.
    """

    # Shared:
    x: torch.Tensor  # [N_total_atoms, NUM_LABELS] one-hot
    y: float  # scalar regression target in [0, 1]
    cluster_of_node: (
        torch.Tensor
    )  # [N_total_atoms] int, which cluster each node belongs to
    K: int  # number of clusters
    N: int  # nodes per cluster

    # Uncut view (single graph with bottleneck):
    edge_index_uncut: torch.Tensor  # [2, E_uncut]
    bottleneck_width: int  # B (for diagnostics; encoded in edge_index_uncut)

    # Cut view (K disconnected graphs):
    edge_index_cut: torch.Tensor  # [2, E_cut], no cross-cluster edges
    # 'batch' under the cut view: each cluster is a separate graph index
    # (0..K-1 within this sample). When packing multiple samples, the
    # collator will offset these.
    batch_cut_local: torch.Tensor  # [N_total_atoms] in 0..K-1


def _random_intra_cluster_edges(
    start_idx: int, N: int, p: float, rng: random.Random
) -> List[Tuple[int, int]]:
    """Erdős–Rényi within a cluster of N nodes starting at index `start_idx`.
    Returns directed edges (both directions for each undirected edge)."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p:
                a, b = start_idx + i, start_idx + j
                edges.append((a, b))
                edges.append((b, a))
    return edges


def _bottleneck_edges(
    cluster_a_start: int,
    cluster_a_size: int,
    cluster_b_start: int,
    cluster_b_size: int,
    B: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Sample exactly B undirected edges between clusters A and B,
    returning them as 2B directed edges (both directions)."""
    edges = []
    chosen = set()
    max_pairs = cluster_a_size * cluster_b_size
    B = min(B, max_pairs)
    while len(chosen) < B:
        i = rng.randint(0, cluster_a_size - 1)
        j = rng.randint(0, cluster_b_size - 1)
        if (i, j) in chosen:
            continue
        chosen.add((i, j))
        a = cluster_a_start + i
        b = cluster_b_start + j
        edges.append((a, b))
        edges.append((b, a))
    return edges


def _generate_one_dumbbell(
    K: int, N: int, B: int, intra_p: float, num_labels: int, rng: random.Random
) -> DumbbellSample:
    """Generate one dumbbell-or-chain sample with K clusters of N nodes,
    bottleneck width B between adjacent clusters."""
    n_total = K * N

    # Node labels (categorical). Used both for features and for the target.
    labels = [rng.randint(0, num_labels - 1) for _ in range(n_total)]
    x = torch.zeros(n_total, num_labels, dtype=torch.float32)
    for ni, lab in enumerate(labels):
        x[ni, lab] = 1.0

    # Cluster assignment per node
    cluster_of_node = torch.tensor(
        [k for k in range(K) for _ in range(N)], dtype=torch.long
    )

    # Intra-cluster edges (same for uncut and cut)
    intra_edges: List[Tuple[int, int]] = []
    for k in range(K):
        intra_edges.extend(_random_intra_cluster_edges(k * N, N, intra_p, rng))

    # Bottleneck edges between adjacent clusters (uncut only)
    bottleneck_edges: List[Tuple[int, int]] = []
    for k in range(K - 1):
        bottleneck_edges.extend(_bottleneck_edges(k * N, N, (k + 1) * N, N, B, rng))

    edge_index_uncut = (
        torch.tensor(intra_edges + bottleneck_edges, dtype=torch.long).t().contiguous()
        if (intra_edges or bottleneck_edges)
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_index_cut = (
        torch.tensor(intra_edges, dtype=torch.long).t().contiguous()
        if intra_edges
        else torch.zeros((2, 0), dtype=torch.long)
    )

    # Target: matching-label pairs between cluster 0 and cluster K-1
    # (the endpoints — for K=2 these are the only two clusters).
    cluster_first_labels = labels[:N]
    cluster_last_labels = labels[(K - 1) * N : K * N]
    matches = 0
    for la in cluster_first_labels:
        for lb in cluster_last_labels:
            if la == lb:
                matches += 1
    y = matches / (N * N)  # in [0, 1]

    # Cut-view local batch (which cluster each node belongs to within this
    # sample). Mirrors cluster_of_node for cut framing.
    batch_cut_local = cluster_of_node.clone()

    return DumbbellSample(
        x=x,
        y=y,
        cluster_of_node=cluster_of_node,
        K=K,
        N=N,
        edge_index_uncut=edge_index_uncut,
        bottleneck_width=B,
        edge_index_cut=edge_index_cut,
        batch_cut_local=batch_cut_local,
    )


def generate_dumbbell_dataset(
    n_samples: int,
    K: int,
    N: int,
    B: int,
    intra_p: float = 0.5,
    num_labels: int = NUM_LABELS,
    seed: int = 0,
) -> List[DumbbellSample]:
    """Generate `n_samples` dumbbell graphs with fixed K, N, B.

    Args:
        n_samples: how many examples to produce
        K: number of clusters (2 = dumbbell, >2 = chain)
        N: nodes per cluster
        B: bottleneck width (edges between adjacent clusters)
        intra_p: edge probability within each cluster (Erdős–Rényi)
        num_labels: vocabulary size for node categorical labels
        seed: RNG seed
    """
    rng = random.Random(seed)
    return [
        _generate_one_dumbbell(
            K=K, N=N, B=B, intra_p=intra_p, num_labels=num_labels, rng=rng
        )
        for _ in range(n_samples)
    ]


# =============================================================================
# Splits and target stats (paralleling other benchmark modules)
# =============================================================================


def random_split(
    items: List[DumbbellSample],
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[DumbbellSample], List[DumbbellSample], List[DumbbellSample]]:
    n = len(items)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    return (
        [items[i] for i in perm[:n_train]],
        [items[i] for i in perm[n_train : n_train + n_val]],
        [items[i] for i in perm[n_train + n_val :]],
    )


@dataclass
class TargetStats:
    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device) -> "TargetStats":
        return TargetStats(self.mean.to(device), self.std.to(device))

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std.clamp_min(1e-9)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


def compute_target_stats(items: List[DumbbellSample]) -> TargetStats:
    if not items:
        raise RuntimeError("empty dataset")
    Y = torch.tensor([it.y for it in items], dtype=torch.float32)
    return TargetStats(mean=Y.mean(), std=Y.std())
