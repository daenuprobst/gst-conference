"""
eval_setaug_synth.py
====================

Set-input training augmentation benchmark on a synthetic planted-motif
classification task.

Research question
-----------------
For a SINGLE-GRAPH classification task, does training the model on sets of
same-class graphs (|S|=k) regularize the backbone enough to improve
SINGLE-GRAPH inference (|S|=1) compared to standard training? And if so,
is GST's cross-graph attention doing something beyond what SupCon does?

Task
----
4-class graph classification. Each graph is a 20-node Erdős-Rényi graph
(p=0.15) with a planted 4-node induced subgraph (motif) characteristic
of the class:
    class 0: K_4   (4-clique)
    class 1: C_4   (4-cycle)
    class 2: K_1,3 (star)
    class 3: P_4   (path)

Node features: one-hot degree (clamped to [0, 8]).

The base motifs have distinguishable degree multisets, but they live
inside an ER noise background and contribute only 4 of 20 nodes; the
model has to identify the motif amid noise. Label noise is added at 8%.

Five conditions
---------------
    A: GST                  trained on single graphs (|S|=1)
    B: GST                  trained on same-class sets (|S|=k=4)
    C: GCN + SupCon aux     trained on single graphs (|S|=1)
    D: GCN + DeepSets head  trained on same-class sets (|S|=k=4)
    E: GCN                  trained on single graphs (|S|=1)

Inference is always single-graph (|S|=1).

Comparisons
-----------
    B vs A : effect of set-input training, GST architecture
    B vs C : effect of set-input training vs SupCon regularization
    B vs D : effect of GST's cross-graph attention beyond simple set pooling
    D vs E : effect of set-input training, GCN architecture
    C vs E : effect of SupCon regularization

Stopping rule (decided before running)
--------------------------------------
For the regularization story to hold, condition B must beat both A and C
with Cohen's d_z > 0.5 and Holm-adjusted p < 0.05 across seeds. Otherwise
the story is falsified.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_set_transformer.models.gst import GraphSetConv


# =============================================================================
# Constants
# =============================================================================
N_NODES = 20
ER_P = 0.15
MOTIF_SIZE = 4
MAX_DEG = 8
NUM_CLASSES = 4
FEATURE_DIM = MAX_DEG + 1  # one-hot degree 0..MAX_DEG

MODEL_KEYS = {
    "A": "GST single",
    "B": "GST set-aug (fixed k)",
    "C": "GCN+SupCon",
    "D": "GCN+DS set-aug (fixed k)",
    "E": "GCN baseline",
    "F": "GST set-aug (var k 1..K)",
    "G": "GCN+DS set-aug (var k 1..K)",
    "H": "GST + node-del subgraphs",
    "I": "GCN+DS + node-del subgraphs",
}


# =============================================================================
# Synthetic data generation
# =============================================================================
def _make_motif_edges(motif_idx: int) -> List[Tuple[int, int]]:
    """Edges of the 4-node motif (local indices 0-3)."""
    if motif_idx == 0:  # K_4
        return [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    if motif_idx == 1:  # C_4
        return [(0, 1), (1, 2), (2, 3), (3, 0)]
    if motif_idx == 2:  # K_1,3 star with center=0
        return [(0, 1), (0, 2), (0, 3)]
    if motif_idx == 3:  # P_4 path 0-1-2-3
        return [(0, 1), (1, 2), (2, 3)]
    raise ValueError(f"Unknown motif: {motif_idx}")


@dataclass
class MotifGraph:
    """One synthetic graph instance."""
    x: torch.Tensor          # [N, F]
    edge_index: torch.Tensor # [2, E]
    y: int                   # class label


def make_planted_graph(rng: np.random.Generator, motif_idx: int) -> MotifGraph:
    """Sample an ER graph then rewire 4 random nodes to form the motif."""
    n = N_NODES
    adj = (rng.random((n, n)) < ER_P).astype(np.int8)
    adj = np.triu(adj, k=1)
    adj = adj + adj.T

    # Pick 4 random nodes for the motif.
    motif_nodes = rng.choice(n, size=MOTIF_SIZE, replace=False)
    # Clear all edges within the chosen 4 nodes.
    for i in range(MOTIF_SIZE):
        for j in range(MOTIF_SIZE):
            adj[motif_nodes[i], motif_nodes[j]] = 0
    # Plant the motif edges.
    for u, v in _make_motif_edges(motif_idx):
        a, b = int(motif_nodes[u]), int(motif_nodes[v])
        adj[a, b] = 1
        adj[b, a] = 1

    # Build edge_index.
    src, dst = np.where(adj > 0)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # Degree-bucketed features.
    deg = adj.sum(axis=1).astype(np.int64).clip(0, MAX_DEG)
    x_idx = torch.tensor(deg, dtype=torch.long)
    x = F.one_hot(x_idx, num_classes=FEATURE_DIM).float()

    return MotifGraph(x=x, edge_index=edge_index, y=motif_idx)


def build_dataset(
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    label_noise: float,
) -> Tuple[List[MotifGraph], List[MotifGraph], List[MotifGraph]]:
    """Stratified across classes; label noise applied only to train."""
    rng = np.random.default_rng(seed)
    total = n_train + n_val + n_test
    per_class = total // NUM_CLASSES
    pool: List[MotifGraph] = []
    for c in range(NUM_CLASSES):
        for _ in range(per_class):
            pool.append(make_planted_graph(rng, c))
    rng.shuffle(pool)
    train = pool[:n_train]
    val = pool[n_train : n_train + n_val]
    test = pool[n_train + n_val : n_train + n_val + n_test]

    # Apply label noise to train only.
    if label_noise > 0:
        for g in train:
            if rng.random() < label_noise:
                # Flip to a different class.
                alt = int(rng.integers(0, NUM_CLASSES - 1))
                if alt >= g.y:
                    alt += 1
                g.y = alt
    return train, val, test


# =============================================================================
# Batching
# =============================================================================
@dataclass
class GraphBatch:
    x: torch.Tensor          # [N_total, F]
    edge_index: torch.Tensor # [2, E_total]
    batch: torch.Tensor      # [N_total] -> graph_id
    set_batch: torch.Tensor  # [G_total] -> set_id
    y: torch.Tensor          # [num_sets]

    def to(self, device):
        return GraphBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            set_batch=self.set_batch.to(device),
            y=self.y.to(device),
        )


def collate(set_groups: List[List[MotifGraph]]) -> GraphBatch:
    """Each entry in set_groups is a list of K graphs sharing a class."""
    xs, eis, batch_idx, set_batch_idx, ys = [], [], [], [], []
    g_offset, n_offset = 0, 0
    for s_idx, group in enumerate(set_groups):
        ys.append(group[0].y)
        for g in group:
            xs.append(g.x)
            eis.append(g.edge_index + n_offset)
            batch_idx.append(torch.full((g.x.size(0),), g_offset, dtype=torch.long))
            set_batch_idx.append(s_idx)
            g_offset += 1
            n_offset += g.x.size(0)
    return GraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long),
        batch=torch.cat(batch_idx, dim=0),
        set_batch=torch.tensor(set_batch_idx, dtype=torch.long),
        y=torch.tensor(ys, dtype=torch.long),
    )


def iter_single_batches(data: List[MotifGraph], batch_size: int, rng: random.Random):
    """Each set is a single graph; batch_size = number of sets per batch."""
    idx = list(range(len(data)))
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        sets = [[data[i]] for i in chunk]
        yield collate(sets)


def make_node_deleted_subgraphs(
    g: MotifGraph,
    k_subgraphs: int,
    rng: random.Random,
) -> List[MotifGraph]:
    """Build a bag of k_subgraphs node-deleted subgraphs of g (ESAN ND policy).

    Each subgraph removes one randomly chosen node. Features are RECOMPUTED
    from the subgraph topology (degrees reflect the subgraph, not the original).
    """
    n = g.x.size(0)
    pick = min(k_subgraphs, n)
    drop_nodes = rng.sample(range(n), pick)

    out: List[MotifGraph] = []
    for drop in drop_nodes:
        keep_mask = torch.ones(n, dtype=torch.bool)
        keep_mask[drop] = False
        n_new = int(keep_mask.sum().item())
        # Old-to-new index mapping.
        old_to_new = torch.full((n,), -1, dtype=torch.long)
        old_to_new[keep_mask] = torch.arange(n_new)
        # Filter edges; remap.
        ei = g.edge_index
        if ei.numel() > 0:
            edge_keep = keep_mask[ei[0]] & keep_mask[ei[1]]
            new_ei = ei[:, edge_keep].clone()
            new_ei[0] = old_to_new[new_ei[0]]
            new_ei[1] = old_to_new[new_ei[1]]
        else:
            new_ei = torch.zeros((2, 0), dtype=torch.long)
        # Recompute degree-bucketed features.
        deg = torch.zeros(n_new, dtype=torch.long)
        if new_ei.numel() > 0:
            deg.scatter_add_(0, new_ei[0], torch.ones(new_ei.size(1), dtype=torch.long))
        deg = deg.clamp(0, MAX_DEG)
        new_x = F.one_hot(deg, num_classes=FEATURE_DIM).float()
        out.append(MotifGraph(x=new_x, edge_index=new_ei, y=g.y))
    return out


def iter_subgraph_batches(
    data: List[MotifGraph],
    batch_size: int,
    k_subgraphs: int,
    rng: random.Random,
):
    """Each input graph becomes a SET of node-deleted subgraphs.
    batch_size = number of input graphs per batch (each contributes k_subgraphs).
    """
    order = list(range(len(data)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        sets: List[List[MotifGraph]] = []
        for idx in chunk:
            bag = make_node_deleted_subgraphs(data[idx], k_subgraphs, rng)
            sets.append(bag)
        yield collate(sets)


def iter_set_batches(
    data: List[MotifGraph],
    batch_size: int,
    k: int,
    rng: random.Random,
    variable_k: bool = False,
    k_min: int = 1,
):
    """Each anchor graph forms a set with same-class siblings.

    If variable_k=False: every set has exactly k graphs.
    If variable_k=True: each set's size is sampled uniformly from [k_min, k].
    Set compositions are resampled every call (i.e. every epoch).

    batch_size = number of sets per batch.
    """
    by_class: Dict[int, List[int]] = {c: [] for c in range(NUM_CLASSES)}
    for i, g in enumerate(data):
        by_class[g.y].append(i)
    anchor_order = list(range(len(data)))
    rng.shuffle(anchor_order)
    for start in range(0, len(anchor_order), batch_size):
        chunk = anchor_order[start : start + batch_size]
        sets: List[List[MotifGraph]] = []
        for anchor_idx in chunk:
            anchor = data[anchor_idx]
            siblings = by_class[anchor.y]
            if variable_k:
                k_actual = rng.randint(k_min, k)
            else:
                k_actual = k
            n_siblings = max(0, k_actual - 1)
            if len(siblings) <= n_siblings:
                picks = siblings.copy()
            else:
                picks = rng.sample(siblings, n_siblings)
            members = [anchor] + [data[i] for i in picks]
            sets.append(members[:k_actual])
        yield collate(sets)


# =============================================================================
# Pure-torch scatter
# =============================================================================
def _scatter_mean(x: torch.Tensor, idx: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Mean of rows of `x` grouped by `idx` into `n_groups` buckets."""
    out = x.new_zeros((n_groups, x.size(-1)))
    out.index_add_(0, idx, x)
    counts = torch.zeros(n_groups, device=x.device, dtype=x.dtype)
    counts.index_add_(0, idx, torch.ones_like(idx, dtype=x.dtype))
    counts = counts.clamp_min(1.0).unsqueeze(-1)
    return out / counts


def _scatter_sum(x: torch.Tensor, idx: torch.Tensor, n_groups: int) -> torch.Tensor:
    out = x.new_zeros((n_groups, x.size(-1)))
    out.index_add_(0, idx, x)
    return out


# =============================================================================
# Models
# =============================================================================
class GCNStack(nn.Module):
    """Plain GCN trunk: GCNConv + LayerNorm + SiLU + residual."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(hidden, hidden, improved=True))
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for gcn, ln, dp in zip(self.layers, self.norms, self.dropouts):
            h = gcn(x, edge_index)
            h = F.silu(ln(h))
            h = dp(h)
            x = x + h
        return x


class GSTStack(nn.Module):
    """Stack of GraphSetConv blocks (broadcast or cross-attn)."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        node_set_mode: str = "broadcast",
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                GraphSetConv(
                    filters=hidden,
                    in_channels=hidden,
                    num_heads=num_heads,
                    mhsa_dropout=dropout,
                    ffn_dropout=dropout,
                    node_set_mode=node_set_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, edge_index, batch, set_batch):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x, edge_index, batch, set_batch)
        return x


class MotifClassifier(nn.Module):
    """Wraps an encoder; supports single-graph and set-input training/inference.

    Conditions A, B use GST encoder; C, E use GCN encoder; D uses GCN encoder
    with a DeepSets pooling over the set during set-input training.
    Condition C adds a SupCon projection head used during training only.

    Forward returns:
        logits: [num_sets, NUM_CLASSES]
        per_graph_emb: [num_graphs, hidden] (used by SupCon)
    """

    def __init__(
        self,
        condition: str,
        hidden: int = 32,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        supcon_proj_dim: int = 64,
    ):
        super().__init__()
        assert condition in MODEL_KEYS
        self.condition = condition
        self.hidden = hidden

        if condition in ("A", "B", "F", "H"):
            self.encoder = GSTStack(
                in_dim=FEATURE_DIM,
                hidden=hidden,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                node_set_mode="broadcast",
            )
            self.is_gst = True
        else:
            self.encoder = GCNStack(
                in_dim=FEATURE_DIM,
                hidden=hidden,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.is_gst = False

        # Per-graph readout (LayerNorm + linear classifier).
        self.graph_norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

        # SupCon projection head — used only when condition == 'C'.
        if condition == "C":
            self.supcon_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, supcon_proj_dim),
            )

        # DeepSets head for D, G, I: pool per-graph embeddings within a set.
        if condition in ("D", "G", "I"):
            self.set_proj_pre = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.set_proj_post = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )

    def encode(self, b: GraphBatch):
        if self.is_gst:
            h = self.encoder(b.x, b.edge_index, b.batch, b.set_batch)
        else:
            h = self.encoder(b.x, b.edge_index)
        # Per-graph mean-pool over nodes.
        n_graphs = int(b.batch.max().item()) + 1 if b.batch.numel() > 0 else 0
        z_graph = _scatter_mean(h, b.batch, n_graphs)
        z_graph = self.graph_norm(z_graph)
        return z_graph

    def forward(self, b: GraphBatch):
        z_graph = self.encode(b)

        # Aggregate per-graph -> per-set, based on condition.
        n_sets = int(b.set_batch.max().item()) + 1 if b.set_batch.numel() > 0 else 0

        if self.condition in ("D", "G", "I"):
            # DeepSets head: phi(g) -> sum-pool -> rho.
            phi = self.set_proj_pre(z_graph)
            pooled = _scatter_sum(phi, b.set_batch, n_sets)
            z_set = self.set_proj_post(pooled)
        else:
            # GST and GCN paths: mean-pool per-graph embeddings within the set.
            # For |S|=1 this is just the per-graph embedding.
            z_set = _scatter_mean(z_graph, b.set_batch, n_sets)

        logits = self.classifier(z_set)
        return logits, z_graph


# =============================================================================
# SupCon loss
# =============================================================================
def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Khosla et al. 2020 supervised contrastive loss.

    z: [N, D] embeddings (assumed L2-normalized).
    labels: [N] integer class labels.
    """
    n = z.size(0)
    if n < 2:
        return z.new_zeros(())
    sim = (z @ z.t()) / temp  # [N, N]
    # Mask out self-similarity.
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    # log softmax row-wise.
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # Positive mask: same-class, not self.
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)
    pos_counts = pos.sum(dim=1).clamp_min(1).float()
    # Average log-prob over positives, then negate and mean.
    per_anchor = -(log_prob * pos.float()).sum(dim=1) / pos_counts
    # Only count anchors with at least one positive.
    valid = pos.any(dim=1)
    if valid.sum() == 0:
        return z.new_zeros(())
    return per_anchor[valid].mean()


# =============================================================================
# Training
# =============================================================================
def evaluate(
    model: MotifClassifier,
    data: List[MotifGraph],
    device: str,
    use_subgraphs: bool = False,
    k_subgraphs: int = 8,
) -> Dict[str, float]:
    """Evaluate accuracy and macro-F1. Single-graph input by default; subgraph
    bag if use_subgraphs=True. Eval-time RNG is fixed for determinism."""
    model.eval()
    rng = random.Random(12345)
    preds, ys = [], []
    with torch.no_grad():
        if use_subgraphs:
            it = iter_subgraph_batches(data, batch_size=16, k_subgraphs=k_subgraphs, rng=rng)
        else:
            it = iter_single_batches(data, batch_size=64, rng=rng)
        for batch in it:
            batch = batch.to(device)
            logits, _ = model(batch)
            pred = logits.argmax(dim=-1)
            preds.append(pred.cpu())
            ys.append(batch.y.cpu())
    p = torch.cat(preds).numpy()
    y = torch.cat(ys).numpy()
    acc = float((p == y).mean())
    # Macro F1.
    f1s = []
    for c in range(NUM_CLASSES):
        tp = float(((p == c) & (y == c)).sum())
        fp = float(((p == c) & (y != c)).sum())
        fn = float(((p != c) & (y == c)).sum())
        if tp == 0:
            f1s.append(0.0)
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec))
    return {"acc": acc, "f1_macro": float(np.mean(f1s))}


def train_one(
    condition: str,
    train_data: List[MotifGraph],
    val_data: List[MotifGraph],
    test_data: List[MotifGraph],
    args,
    seed: int,
    device: str,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed * 9973 + 1)

    model = MotifClassifier(
        condition=condition,
        hidden=args.hidden,
        num_layers=args.layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_sets = condition in ("B", "D", "F", "G")
    use_subgraphs = condition in ("H", "I")
    use_supcon = condition == "C"
    variable_k = condition in ("F", "G")

    best_val = -float("inf")
    best_state = None
    patience = args.patience
    no_improve = 0
    t0 = time.time()

    history = []
    for epoch in range(args.epochs):
        model.train()
        if use_sets:
            it = iter_set_batches(
                train_data,
                batch_size=args.batch_size_sets,
                k=args.k,
                rng=rng,
                variable_k=variable_k,
                k_min=args.k_min,
            )
        elif use_subgraphs:
            it = iter_subgraph_batches(
                train_data,
                batch_size=args.batch_size_subgraphs,
                k_subgraphs=args.k_subgraphs,
                rng=rng,
            )
        else:
            it = iter_single_batches(
                train_data, batch_size=args.batch_size_single, rng=rng
            )

        ep_loss, n_batches = 0.0, 0
        for batch in it:
            batch = batch.to(device)
            logits, z_graph = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            if use_supcon:
                z = F.normalize(model.supcon_proj(z_graph), dim=-1)
                per_graph_labels = batch.y[batch.set_batch]
                loss = loss + args.supcon_weight * supcon_loss(z, per_graph_labels, temp=args.supcon_temp)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1

        # Eval on val.
        val_metrics = evaluate(
            model, val_data, device,
            use_subgraphs=use_subgraphs, k_subgraphs=args.k_subgraphs,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": ep_loss / max(1, n_batches),
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1_macro"],
            }
        )
        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(
        model, test_data, device,
        use_subgraphs=use_subgraphs, k_subgraphs=args.k_subgraphs,
    )
    elapsed = time.time() - t0
    return {
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["f1_macro"],
        "best_val_acc": best_val,
        "params": n_params,
        "elapsed_s": elapsed,
        "epochs_run": len(history),
        "history": history,
    }


# =============================================================================
# Stats helpers
# =============================================================================
def t_ci_95(values):
    if len(values) < 2:
        return float("nan")
    from scipy.stats import t as t_dist

    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    half = float(t_dist.ppf(0.975, df=len(arr) - 1)) * sem
    return mean, half


def cohens_d_paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return float("nan")
    return float(diff.mean() / diff.std(ddof=1))


def wilcoxon_signed_rank(a, b):
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    from scipy.stats import wilcoxon

    try:
        return float(wilcoxon(a, b, zero_method="zsplit").pvalue)
    except Exception:
        return float("nan")


def holm_bonferroni(pvalues, alpha=0.05):
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adj = [None] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        scale = n - rank
        v = min(1.0, pvalues[i] * scale)
        v = max(v, running_max)
        adj[i] = v
        running_max = v
    return adj


# =============================================================================
# Reporting
# =============================================================================
def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)


def render_summary(per_cond: Dict[str, Dict], comparisons: List[Dict]) -> str:
    lines = []
    lines.append("# eval_setaug_synth results\n")
    lines.append("## Per-condition test metrics (mean ± 95% CI over seeds)\n")
    lines.append("| cond | name | params | test acc | test F1 (macro) | best val acc |")
    lines.append("|---|---|---:|---|---|---|")
    for c in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        if c not in per_cond:
            continue
        s = per_cond[c]
        m, h = s["acc_mean_ci"]
        m_f1, h_f1 = s["f1_mean_ci"]
        lines.append(
            f"| {c} | {MODEL_KEYS[c]} | {s['params']:,} | "
            f"{m:.4f} ± {h:.4f} | {m_f1:.4f} ± {h_f1:.4f} | "
            f"{s['best_val_mean']:.4f} |"
        )
    lines.append("\n## Paired comparisons on test accuracy (Holm-adjusted)\n")
    lines.append("| comparison | mean diff | Cohen's d_z | p (adj) | reject H0 |")
    lines.append("|---|---:|---:|---:|:---:|")
    for cmp in comparisons:
        lines.append(
            f"| {cmp['a']} vs {cmp['b']} | {cmp['mean_diff']:+.4f} | "
            f"{cmp['dz']:+.2f} | {cmp['p_adj']:.3f} | "
            f"{'yes' if cmp['reject'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="results/setaug_synth")
    p.add_argument("--quick", action="store_true")

    # Data
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--label-noise", type=float, default=0.08)
    p.add_argument("--data-seed", type=int, default=0)

    # Architecture
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size-single", type=int, default=32)
    p.add_argument("--batch-size-sets", type=int, default=8)
    p.add_argument("--batch-size-subgraphs", type=int, default=8, help="Inputs per batch for H/I (each contributes k_subgraphs subgraphs).")
    p.add_argument("--k-subgraphs", type=int, default=8, help="Number of node-deleted subgraphs per input for H/I.")
    p.add_argument("--k", type=int, default=4, help="Fixed k for B/D; max k for F/G when variable.")
    p.add_argument("--k-min", type=int, default=1, help="Min k for F/G variable-k mode.")
    p.add_argument("--supcon-weight", type=float, default=0.5)
    p.add_argument("--supcon-temp", type=float, default=0.1)

    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
    )

    args = p.parse_args()
    if args.quick:
        args.epochs = 5
        args.patience = 3
        args.seeds = 1
        args.n_train = 200
        args.n_val = 100
        args.n_test = 200
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building dataset (train={args.n_train}, val={args.n_val}, test={args.n_test}, noise={args.label_noise})...")
    train, val, test = build_dataset(
        seed=args.data_seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        label_noise=args.label_noise,
    )
    print(f"  class balance (train): {[sum(1 for g in train if g.y == c) for c in range(NUM_CLASSES)]}")
    print(f"  class balance (test):  {[sum(1 for g in test if g.y == c) for c in range(NUM_CLASSES)]}")

    raw_rows = []
    per_cond: Dict[str, Dict] = {}
    for cond in args.conditions:
        print(f"\n=== Condition {cond}: {MODEL_KEYS[cond]} ===")
        accs, f1s, val_accs, params_list, times = [], [], [], [], []
        for sidx in range(args.seeds):
            seed = args.seed_offset + sidx
            res = train_one(cond, train, val, test, args, seed, args.device)
            print(
                f"  seed={seed}  test_acc={res['test_acc']:.4f}  "
                f"test_f1={res['test_f1']:.4f}  best_val={res['best_val_acc']:.4f}  "
                f"({res['elapsed_s']:.1f}s, {res['params']:,} params, epochs={res['epochs_run']})"
            )
            accs.append(res["test_acc"])
            f1s.append(res["test_f1"])
            val_accs.append(res["best_val_acc"])
            params_list.append(res["params"])
            times.append(res["elapsed_s"])
            raw_rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "test_acc": res["test_acc"],
                    "test_f1": res["test_f1"],
                    "best_val_acc": res["best_val_acc"],
                    "params": res["params"],
                    "elapsed_s": res["elapsed_s"],
                    "epochs_run": res["epochs_run"],
                }
            )

        ci = t_ci_95(accs)
        ci_f1 = t_ci_95(f1s)
        per_cond[cond] = {
            "name": MODEL_KEYS[cond],
            "params": int(np.mean(params_list)),
            "test_acc_seeds": accs,
            "test_f1_seeds": f1s,
            "best_val_seeds": val_accs,
            "acc_mean_ci": ci if isinstance(ci, tuple) else (float(np.mean(accs)), float("nan")),
            "f1_mean_ci": ci_f1 if isinstance(ci_f1, tuple) else (float(np.mean(f1s)), float("nan")),
            "best_val_mean": float(np.mean(val_accs)),
            "total_train_seconds": float(sum(times)),
        }

    # Paired comparisons on test accuracy.
    comparisons_pairs = [
        # Fixed-k vs reference
        ("B", "A"),
        ("B", "C"),
        ("B", "D"),
        ("B", "E"),
        ("D", "E"),
        ("C", "E"),
        ("A", "E"),
        # Variable-k vs everything that matters
        ("F", "A"),
        ("F", "B"),
        ("F", "C"),
        ("F", "E"),
        ("G", "D"),
        ("G", "E"),
        ("F", "G"),
        # Subgraph reframing
        ("H", "A"),
        ("H", "E"),
        ("H", "I"),
        ("I", "E"),
    ]
    raw_p = []
    cmp_meta = []
    for a, b in comparisons_pairs:
        if a not in per_cond or b not in per_cond:
            continue
        sa = per_cond[a]["test_acc_seeds"]
        sb = per_cond[b]["test_acc_seeds"]
        dz = cohens_d_paired(sa, sb)
        p = wilcoxon_signed_rank(sa, sb)
        raw_p.append(p)
        cmp_meta.append({"a": a, "b": b, "mean_diff": float(np.mean(sa)) - float(np.mean(sb)), "dz": dz, "p": p})
    adj = holm_bonferroni(raw_p)
    comparisons = []
    for cmp, pa in zip(cmp_meta, adj):
        cmp = dict(cmp)
        cmp["p_adj"] = pa
        cmp["reject"] = (cmp["dz"] > 0.5 and pa < 0.05) if not math.isnan(cmp["dz"]) else False
        comparisons.append(cmp)

    write_json(out_dir / "raw.json", {"per_cond": per_cond, "comparisons": comparisons, "rows": raw_rows, "args": vars(args)})
    (out_dir / "summary.md").write_text(render_summary(per_cond, comparisons))
    print(f"\nWrote {out_dir / 'raw.json'} and {out_dir / 'summary.md'}")

    # Final verdict.
    print("\n========= verdict =========")
    for label, a, b in [
        ("B vs A (fixed-k set-aug vs single)", "B", "A"),
        ("B vs C (fixed-k set-aug vs SupCon)", "B", "C"),
        ("F vs A (var-k set-aug vs single)", "F", "A"),
        ("F vs B (var-k vs fixed-k)", "F", "B"),
        ("F vs C (var-k set-aug vs SupCon)", "F", "C"),
        ("G vs D (var-k vs fixed-k DS)", "G", "D"),
        ("H vs A (GST+subgraphs vs GST single)", "H", "A"),
        ("H vs I (GST attn vs DS over same subgraphs)", "H", "I"),
        ("H vs E (GST+subgraphs vs GCN)", "H", "E"),
        ("I vs E (DS+subgraphs vs GCN)", "I", "E"),
    ]:
        cmp = next((c for c in comparisons if c["a"] == a and c["b"] == b), None)
        if cmp:
            print(
                f"  {label}: d_z={cmp['dz']:+.2f}, p_adj={cmp['p_adj']:.3f}, "
                f"reject={cmp['reject']}"
            )


if __name__ == "__main__":
    main()
