"""
synth_benchmark.py
==================

Comprehensive synthetic benchmark for *set-of-graphs* architectures.

Compares four models on synthetic tasks designed to probe set-conditional
structural reasoning — the inductive bias the GraphSetConv block was designed
to exploit:

    - GraphSetConv-CrossAttn   (interleaved, per-node cross-attention to set tokens)
    - GraphSetConv-Broadcast   (interleaved, uniform broadcast back to nodes)
    - GCN+DeepSets             (pipeline: GCN encoder, then permutation-invariant pool)
    - GCN+SetTransformer       (pipeline: GCN encoder, then set-level self-attention)

Tasks
-----
1. RING K-hop classification (primary diagnostic)
       y(v) = 1  iff  dist(v, nearest set-anchor) == K
       set-anchor = node attaining the SET-WIDE max degree.
       Constant positive density across K, no saturation; forces the model to
       (a) extract a set-wide statistic, (b) propagate it back through edges,
       (c) encode exact distance. Metric: Average Precision (AP).

2. BALL K-hop classification (saturating control)
       y(v) = 1  iff  dist(v, nearest set-anchor) <= K
       Recovers the original "K-hop reachability" formulation. Saturates
       quickly, included to verify the architecture comparison reproduces
       the known pattern at low K. Metric: Average Precision (AP).

3. SET-ANCHOR distance regression
       y(v) = min(dist(v, nearest set-anchor in graph), D_max)
       graphs without an anchor: y(v) = D_max for all nodes.
       Continuous analog of the ring task. Metric: MAE on capped distance.

4. SET-SIZE GENERALIZATION (extrapolation)
       Train the ring task with set-size K_set in [k_train_lo, k_train_hi];
       evaluate on held-out K_set values not seen at training time.
       Tests whether the architecture's set-size invariance is robust.
       Metric: AP at each test K_set.

Methodology references
----------------------
- Ring labels & exact-distance probing: You & Ying et al., "Position-aware
  GNNs" (ICML 2019); Zhang & Chen, "Rethinking the Expressive Power of GNNs
  via Graph Biconnectivity" (ICLR 2023; GD-WL).
- Receptive-field / oversquashing framing: Alon & Yahav, "On the Bottleneck
  of GNNs" (ICLR 2021); Topping et al. (ICLR 2022); Dwivedi et al.,
  "Long Range Graph Benchmark" (NeurIPS 2022).
- Long-range synthetic propagation tests: Miglior et al., "Can You Hear Me
  Now? A Benchmark for Long-Range Graph Propagation" (ECHO, 2025).
- Set-of-graphs / nested-bag framing: Tibo et al., "Multi-Multi-Instance
  Learning" (JMLR 2020); Pal et al., "Bag Graph: MIL using Bayesian GNNs" (2022).
- Inductive-bias probing on controlled synthetic data:
  Lake et al., SCAN benchmark (ICML 2018) — established practice.

Statistical reporting
---------------------
- N seeds per cell (default 10).
- Mean +/- std and 95% Student-t confidence intervals.
- Paired Wilcoxon signed-rank test on per-seed scores for headline pairs
  (GraphSetConv-CrossAttn vs each baseline; GraphSetConv-Broadcast vs each
  baseline; CrossAttn vs Broadcast).
- Cohen's d for paired samples as effect size.
- Holm-Bonferroni correction across pairs within a (task, K) cell.

Outputs
-------
results/<run_name>/
    raw.json            -- per-seed scores, params, wall-clock
    summary.csv         -- flat table for re-analysis
    summary.md          -- human-readable
    summary.tex         -- LaTeX booktabs, ready to paste
    pairwise.tex        -- pairwise significance table
    config.json         -- exact CLI args + library versions for reproducibility

Parameter equalization
----------------------
--equalize-params {none, smaller, larger}
    none      : every model uses --hidden as its hidden dim (default).
    smaller   : hidden_dim of GraphSetConv variants is scaled DOWN until both
                their parameter counts match (within tolerance) the smaller
                of {GCN+DeepSets, GCN+SetTransformer}.
    larger    : baseline hidden_dim is scaled UP until parameter counts match
                GraphSetConv at --hidden. (Tests baselines at GSC's budget.)
    Param matching is reported in the run config; both tables show param
    counts alongside scores.

Usage
-----
Quick smoke test (1 seed, low steps, two K values):
    python synth_benchmark.py --quick

Full publication run (default 10 seeds, all tasks, equalized to smaller):
    python synth_benchmark.py --equalize-params smaller --tasks all \\
        --out-dir results/main

Ring task only at default budget:
    python synth_benchmark.py --tasks ring --out-dir results/ring_natural

Just the GSC mode comparison (cross-attn vs broadcast):
    python synth_benchmark.py --models gsc-ca gsc-bc --tasks ring \\
        --out-dir results/gsc_ablation

Requires graph_set_conv.py in the same directory (or on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, mean_absolute_error
from torch_geometric.nn import (
    GCNConv,
    GraphNorm,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch

# Local: GraphSetConv block (the architecture under test).
from graph_set_transformer.models.gst_v2 import GraphSetConv  # noqa: E402


# ============================================================================
# Synthetic data
# ============================================================================


@dataclass
class GraphPreset:
    """ER graph-size presets. The two settings differ in expected diameter,
    which controls how informative each K is for ring/ball labels."""

    name: str
    n_min: int
    n_max: int
    p_min: float
    p_max: float


PRESETS = {
    "default": GraphPreset("default", 12, 25, 0.10, 0.25),  # diam approx 2-3
    "large": GraphPreset("large", 40, 80, 0.05, 0.10),  # diam approx 5-8
}


def sample_er_graph(rng: random.Random, preset: GraphPreset) -> nx.Graph:
    n = rng.randint(preset.n_min, preset.n_max)
    p = rng.uniform(preset.p_min, preset.p_max)
    g = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 2**31 - 1))
    # Avoid edge-empty graphs (GCNConv self-loop fallback would still work,
    # but distances become trivially infinite for everything not the anchor).
    if g.number_of_edges() == 0:
        u, v = rng.sample(range(n), 2)
        g.add_edge(u, v)
    return g


def degrees(g: nx.Graph) -> np.ndarray:
    return np.array([g.degree(i) for i in range(g.number_of_nodes())], dtype=np.int64)


def shortest_path_lengths(g: nx.Graph, sources: Sequence[int]) -> np.ndarray:
    """Min distance from each node to nearest node in `sources`. Disconnected
    nodes get +inf. Returns float array of shape [N]."""
    n = g.number_of_nodes()
    out = np.full(n, np.inf, dtype=np.float32)
    if not sources:
        return out
    # Multi-source BFS via NetworkX: cheaper than per-source loops for the
    # graph sizes we're using, and exact (BFS = shortest path on unweighted).
    for s in sources:
        lengths = nx.single_source_shortest_path_length(g, s)
        for v, dv in lengths.items():
            if dv < out[v]:
                out[v] = dv
    return out


def build_set(
    rng: random.Random, k_set: int, preset: GraphPreset
) -> Tuple[List[nx.Graph], List[np.ndarray], int]:
    """Sample k_set ER graphs and return (graphs, degree_arrays, set_max_degree)."""
    graphs = [sample_er_graph(rng, preset) for _ in range(k_set)]
    degs = [degrees(g) for g in graphs]
    set_max = int(max(d.max() for d in degs))
    return graphs, degs, set_max


def label_ring(
    graphs: List[nx.Graph], degs: List[np.ndarray], set_max: int, k: int
) -> List[np.ndarray]:
    """Per-node binary labels. y[v] = 1 iff dist(v, nearest set-anchor) == k.
    Anchors are nodes whose degree equals set_max (the SET-WIDE max).
    Graphs with no anchor: all-zero labels."""
    out = []
    for g, d in zip(graphs, degs):
        anchors = [int(i) for i, deg_i in enumerate(d) if deg_i == set_max]
        n = g.number_of_nodes()
        y = np.zeros(n, dtype=np.float32)
        if anchors and k == 0:
            for a in anchors:
                y[a] = 1.0
        elif anchors:
            dist = shortest_path_lengths(g, anchors)
            y[dist == k] = 1.0
        out.append(y)
    return out


def label_ball(
    graphs: List[nx.Graph], degs: List[np.ndarray], set_max: int, k: int
) -> List[np.ndarray]:
    """Per-node binary labels. y[v] = 1 iff dist(v, nearest set-anchor) <= k.
    Saturating control task (the K-ball saturates the anchor graph at high K)."""
    out = []
    for g, d in zip(graphs, degs):
        anchors = [int(i) for i, deg_i in enumerate(d) if deg_i == set_max]
        n = g.number_of_nodes()
        y = np.zeros(n, dtype=np.float32)
        if anchors:
            dist = shortest_path_lengths(g, anchors)
            y[dist <= k] = 1.0
        out.append(y)
    return out


def label_distance(
    graphs: List[nx.Graph], degs: List[np.ndarray], set_max: int, d_max: int
) -> List[np.ndarray]:
    """Per-node distance regression. y[v] = min(dist(v, nearest anchor), d_max).
    Graphs without an anchor: y[v] = d_max (saturates upward)."""
    out = []
    for g, d in zip(graphs, degs):
        anchors = [int(i) for i, deg_i in enumerate(d) if deg_i == set_max]
        n = g.number_of_nodes()
        if anchors:
            dist = shortest_path_lengths(g, anchors)
            dist = np.where(np.isinf(dist), d_max, dist)
            y = np.minimum(dist, d_max).astype(np.float32)
        else:
            y = np.full(n, d_max, dtype=np.float32)
        out.append(y)
    return out


# ----------------------------------------------------------------------------
# Batch packing
# ----------------------------------------------------------------------------


def _features_from_degree(d: np.ndarray, in_dim: int) -> np.ndarray:
    """Tiny degree-based input features. Bare degree is leaky for the ring
    task (model could shortcut to "max degree = anchor"), so we deliberately
    expose ONLY a normalized degree and a constant-1 bias. The set-wide
    threshold still has to be inferred from the set."""
    if in_dim < 2:
        raise ValueError("in_dim must be >= 2")
    n = d.shape[0]
    feats = np.zeros((n, in_dim), dtype=np.float32)
    feats[:, 0] = 1.0  # bias
    feats[:, 1] = d.astype(np.float32) / 10  # scaled degree
    if in_dim > 2:
        # Constant-noise extra channels (so depth helps but doesn't reveal
        # additional task signal). Standard probe-task practice.
        feats[:, 2:] = 0.0
    return feats


@dataclass
class SetBatch:
    """One mini-batch of `B` sets, packed as a single PyG-style graph batch."""

    x: torch.Tensor  # [N_total, in_dim]
    edge_index: torch.Tensor  # [2, E_total]
    batch: torch.Tensor  # [N_total] graph id, contiguous from 0
    set_batch: torch.Tensor  # [G_total] set id per graph, contiguous from 0
    y: torch.Tensor  # [N_total] node-level target
    n_sets: int
    n_graphs: int
    n_nodes: int


def collate_to_batch(samples, in_dim: int, device: str) -> SetBatch:
    """Flatten a list of per-set (graphs, labels) into a single batched object."""
    xs, edge_indices, batch_ids, set_ids, ys = [], [], [], [], []
    n_offset, g_offset, s_offset = 0, 0, 0
    for graphs, degs, labels in samples:
        for g, d, y in zip(graphs, degs, labels):
            n = g.number_of_nodes()
            feats = _features_from_degree(d, in_dim)
            xs.append(torch.from_numpy(feats))
            ys.append(torch.from_numpy(y))
            if g.number_of_edges() > 0:
                ei = np.array(list(g.edges)).T  # [2, E]
                ei = np.concatenate([ei, ei[::-1]], axis=1)  # undirected
                ei = torch.from_numpy(ei.astype(np.int64)) + n_offset
            else:
                ei = torch.zeros((2, 0), dtype=torch.long)
            edge_indices.append(ei)
            batch_ids.append(torch.full((n,), g_offset, dtype=torch.long))
            set_ids.append(torch.tensor([s_offset], dtype=torch.long))
            n_offset += n
            g_offset += 1
        s_offset += 1
    x = torch.cat(xs, dim=0).to(device)
    edge_index = torch.cat(edge_indices, dim=1).to(device)
    batch = torch.cat(batch_ids, dim=0).to(device)
    set_batch = torch.cat(set_ids, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    return SetBatch(
        x=x,
        edge_index=edge_index,
        batch=batch,
        set_batch=set_batch,
        y=y,
        n_sets=s_offset,
        n_graphs=g_offset,
        n_nodes=n_offset,
    )


def make_sample_iterator(
    rng: random.Random,
    k_set_range: Tuple[int, int],
    preset: GraphPreset,
    label_fn: Callable,
):
    """Infinite generator of (graphs, degs, labels) tuples for one set."""
    k_lo, k_hi = k_set_range
    while True:
        k_set = rng.randint(k_lo, k_hi)
        graphs, degs, set_max = build_set(rng, k_set, preset)
        labels = label_fn(graphs, degs, set_max)
        yield graphs, degs, labels


def audit_label_density(
    rng_seed: int,
    k_set_range: Tuple[int, int],
    preset: GraphPreset,
    label_fn: Callable,
    n: int = 600,
) -> Tuple[float, float]:
    """Reports (mean positive density, mean fraction with-anchor graphs).
    For binary tasks; for distance regression returns (mean, std) of targets."""
    rng = random.Random(rng_seed)
    pos_total, n_total = 0, 0
    has_anchor_total, g_total = 0, 0
    all_targets = []
    for _ in range(n):
        k_set = rng.randint(*k_set_range)
        graphs, degs, set_max = build_set(rng, k_set, preset)
        labels = label_fn(graphs, degs, set_max)
        for g, d, y in zip(graphs, degs, labels):
            n_total += y.shape[0]
            if y.dtype == np.float32 and set(np.unique(y)).issubset({0.0, 1.0}):
                pos_total += int(y.sum())
            else:
                all_targets.append(y)
            if (d == set_max).any():
                has_anchor_total += 1
            g_total += 1
    if all_targets:
        cat = np.concatenate(all_targets)
        return float(cat.mean()), float(cat.std())
    return pos_total / max(n_total, 1), has_anchor_total / max(g_total, 1)


# ============================================================================
# Models
# ============================================================================


def make_activation():
    return nn.SiLU()


class InputProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)

    def forward(self, x):
        return self.proj(x)


# ----- Pipeline baselines (encoder shared, set head differs) -----


class GCNEncoder(nn.Module):
    """Shared GCN trunk: input proj + L * (GCN + GraphNorm + SiLU + dropout)."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = InputProjection(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden, hidden, improved=True))
            self.norms.append(GraphNorm(hidden))
        self.act = make_activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.in_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h, batch)
            h = self.act(h)
            h = self.dropout(h)
            x = x + h  # pre-act residual; matches GraphSetConv resid pattern
        return x


class DeepSetsHead(nn.Module):
    """Permutation-invariant set encoder: per-graph pool -> MLP -> sum -> MLP."""

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(hidden, hidden),
            make_activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden),
            make_activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, z_graph, set_batch):
        # z_graph: [G, D], one vector per graph; aggregate to set vector.
        h = self.phi(z_graph)
        z_set = global_add_pool(h, set_batch)  # [S, D]
        z_set = self.rho(z_set)  # [S, D]
        return z_set  # [S, D]


class SetTransformerHead(nn.Module):
    """Self-attention over per-graph tokens (a la Lee et al. 2019 SAB block)."""

    def __init__(
        self, hidden: int, num_heads: int, dropout: float = 0.1, ffn_mult: int = 2
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.mha = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_mult),
            make_activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden * ffn_mult, hidden),
            nn.Dropout(dropout),
        )
        # PMA-style readout: a single learnable seed token aggregates the set.
        self.seed = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.ln3 = nn.LayerNorm(hidden)
        self.pma = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, z_graph, set_batch):
        z_dense, mask = to_dense_batch(z_graph, set_batch)  # [S, G_max, D]
        z_n = self.ln1(z_dense)
        attn, _ = self.mha(z_n, z_n, z_n, key_padding_mask=~mask, need_weights=False)
        z_dense = z_dense + attn
        z_dense = z_dense + self.ffn(self.ln2(z_dense))
        # PMA pool to a single set vector
        S = z_dense.shape[0]
        seed = self.seed.expand(S, -1, -1)  # [S, 1, D]
        zn = self.ln3(z_dense)
        z_set, _ = self.pma(seed, zn, zn, key_padding_mask=~mask, need_weights=False)
        return z_set.squeeze(1)  # [S, D]


class PipelineModel(nn.Module):
    """L * GCN, then a set head; node-level prediction concatenates
    [node_emb, graph_emb_at_node, set_emb_at_node] -> MLP. This is the
    deliberately-strong baseline configuration: pipelines are NOT denied
    set context; they just receive it once, at the very end."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_layers: int,
        set_head: str,
        num_heads: int,
        task: str,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden, num_layers, dropout=dropout)
        if set_head == "deepsets":
            self.set_head = DeepSetsHead(hidden, dropout=dropout)
        elif set_head == "settransformer":
            self.set_head = SetTransformerHead(hidden, num_heads, dropout=dropout)
        else:
            raise ValueError(set_head)
        # Per-node head sees node, its graph context, and its set context.
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            make_activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.task = task

    def forward(self, x, edge_index, batch, set_batch):
        h = self.encoder(x, edge_index, batch)
        z_graph = global_mean_pool(h, batch)  # [G, D]
        z_set = self.set_head(z_graph, set_batch)  # [S, D]
        graph_at_node = z_graph[batch]  # [N, D]
        set_at_node = z_set[set_batch[batch]]  # [N, D]
        feat = torch.cat([h, graph_at_node, set_at_node], dim=-1)
        return self.head(feat).squeeze(-1)  # [N]


# ----- GraphSetConv wrapper -----


class GraphSetConvModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_layers: int,
        num_heads: int,
        node_set_mode: str,
        task: str,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert node_set_mode in ("cross_attn", "broadcast"), node_set_mode
        self.in_proj = InputProjection(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                GraphSetConv(
                    filters=hidden,
                    in_channels=hidden,
                    activation="silu",
                    mhsa_dropout=dropout,
                    ffn_dropout=dropout,
                    pooling="attn",
                    use_gating=True,
                    ffn_multiplier=2,
                    num_heads=num_heads,
                    drop_path=0.0,
                    node_set_mode=node_set_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            make_activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.task = task

    def forward(self, x, edge_index, batch, set_batch):
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x, edge_index, batch, set_batch)
        return self.head(x).squeeze(-1)


# ----------------------------------------------------------------------------
# Builder registry
# ----------------------------------------------------------------------------


MODEL_KEYS = {
    "gsc-ca": "GraphSetConv-CrossAttn",
    "gsc-bc": "GraphSetConv-Broadcast",
    "gcn-ds": "GCN+DeepSets",
    "gcn-st": "GCN+SetTransformer",
}


def build_model(
    key: str,
    in_dim: int,
    hidden: int,
    num_layers: int,
    num_heads: int,
    task: str,
    dropout: float = 0.1,
    num_pool_tokens: int = 4,
    num_registers: int = 2,
) -> nn.Module:
    if key == "gsc-ca":
        return GraphSetConvModel(
            in_dim, hidden, num_layers, num_heads, "cross_attn", task, dropout=dropout
        )
    if key == "gsc-bc":
        return GraphSetConvModel(
            in_dim, hidden, num_layers, num_heads, "broadcast", task, dropout=dropout
        )
    if key == "gcn-ds":
        return PipelineModel(
            in_dim, hidden, num_layers, "deepsets", num_heads, task, dropout=dropout
        )
    if key == "gcn-st":
        return PipelineModel(
            in_dim,
            hidden,
            num_layers,
            "settransformer",
            num_heads,
            task,
            dropout=dropout,
        )
    raise ValueError(key)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Parameter equalization
# ============================================================================


def find_hidden_for_target_params(
    builder: Callable[[int], nn.Module],
    target: int,
    num_heads: int,
    tol_frac: float = 0.05,
    lo: int = 16,
    hi: int = 1024,
) -> Tuple[int, int]:
    """Binary-search / grid-search hidden_dim so that builder(hidden) yields
    a model with parameter count within tol_frac of target. hidden must be
    divisible by num_heads. Returns (best_hidden, params_at_best)."""
    best_h, best_p, best_err = None, None, float("inf")
    candidates = [h for h in range(lo, hi + 1) if h % num_heads == 0]
    # Coarse pass first to bracket, then refine.
    for h in candidates:
        m = builder(h)
        p = count_params(m)
        del m
        err = abs(p - target)
        if err < best_err:
            best_err, best_h, best_p = err, h, p
        if p > target * 4:  # too big, stop
            break
    return best_h, best_p


def equalize_models(
    in_dim: int,
    hidden: int,
    num_layers: int,
    num_heads: int,
    task: str,
    mode: str,
    num_pool_tokens: int = 4,
    num_registers: int = 2,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Resolve per-model hidden-dim and parameter count under equalization.

    mode 'none'    : every model uses `hidden`.
    mode 'smaller' : GSC variants scaled down to match smaller-baseline.
    mode 'larger'  : baselines scaled up to match GSC at `hidden`.

    Returns (hiddens_by_model_key, params_by_model_key)."""
    base_hiddens = {k: hidden for k in MODEL_KEYS}
    base_params = {
        k: count_params(
            build_model(
                k,
                in_dim,
                hidden,
                num_layers,
                num_heads,
                task,
                num_pool_tokens=num_pool_tokens,
                num_registers=num_registers,
            )
        )
        for k in MODEL_KEYS
    }
    if mode == "none":
        return base_hiddens, base_params

    def builder_for(key):
        return lambda h: build_model(
            key,
            in_dim,
            h,
            num_layers,
            num_heads,
            task,
            num_pool_tokens=num_pool_tokens,
            num_registers=num_registers,
        )

    if mode == "smaller":
        target_key = min(["gcn-ds", "gcn-st"], key=lambda k: base_params[k])
        target_p = base_params[target_key]
        hiddens = {k: base_hiddens[k] for k in ["gcn-ds", "gcn-st"]}
        params = {k: base_params[k] for k in ["gcn-ds", "gcn-st"]}
        for k in MODEL_KEYS:
            if k in ("gcn-ds", "gcn-st"):
                continue
            h, p = find_hidden_for_target_params(builder_for(k), target_p, num_heads)
            hiddens[k] = h
            params[k] = p
        return hiddens, params

    if mode == "larger":
        gsc_keys = [k for k in MODEL_KEYS if k.startswith("gsc-")]
        target_p = max(base_params[k] for k in gsc_keys)
        hiddens = {k: base_hiddens[k] for k in gsc_keys}
        params = {k: base_params[k] for k in gsc_keys}
        for k in ["gcn-ds", "gcn-st"]:
            h, p = find_hidden_for_target_params(builder_for(k), target_p, num_heads)
            hiddens[k] = h
            params[k] = p
        return hiddens, params

    raise ValueError(f"Unknown equalization mode: {mode}")


# ============================================================================
# Training and evaluation
# ============================================================================


@dataclass
class CellSpec:
    """One cell of the experiment grid."""

    task: str  # "ring" | "ball" | "distance" | "ring_extrapolate"
    k_value: int  # ring/ball K, or train upper bound for extrapolate
    test_k_set: Optional[int] = None  # for ring_extrapolate: set-size at eval
    extra: Dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        if self.task == "ring_extrapolate":
            return f"ring_extrapolate_K{self.k_value}_set{self.test_k_set}"
        return f"{self.task}_K{self.k_value}"


def make_label_fn(task: str, k_value: int, d_max: int):
    if task == "ring" or task == "ring_extrapolate":

        def lab(graphs, degs, sm):
            return label_ring(graphs, degs, sm, k_value)

        return lab
    if task == "ball":

        def lab(graphs, degs, sm):
            return label_ball(graphs, degs, sm, k_value)

        return lab
    if task == "distance":

        def lab(graphs, degs, sm):
            return label_distance(graphs, degs, sm, d_max)

        return lab
    raise ValueError(task)


def loss_for_task(task: str):
    if task in ("ring", "ball", "ring_extrapolate"):
        return F.binary_cross_entropy_with_logits
    if task == "distance":
        # Smooth-L1: robust to outliers from the d_max-clipped tail.
        return F.smooth_l1_loss
    raise ValueError(task)


def metric_for_task(task: str):
    if task in ("ring", "ball", "ring_extrapolate"):

        def ap(y_true, y_score):
            # AP undefined when only one class present -> NaN handled upstream
            if len(set(y_true.tolist())) < 2:
                return float("nan")
            return float(average_precision_score(y_true, y_score))

        return "AP", ap
    if task == "distance":

        def mae(y_true, y_pred):
            return float(mean_absolute_error(y_true, y_pred))

        return "MAE", mae
    raise ValueError(task)


def train_one(
    model: nn.Module,
    cell: CellSpec,
    args,
    seed: int,
    in_dim: int,
    preset: GraphPreset,
    device: str,
) -> Tuple[float, float, Dict]:
    """Train and evaluate one (model, cell, seed). Returns (test_metric,
    elapsed_seconds, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_rng = random.Random(seed * 7919 + 1)
    val_rng = random.Random(seed * 7919 + 2)
    test_rng = random.Random(seed * 7919 + 3)

    train_k_range = tuple(args.k_set_range)
    if cell.task == "ring_extrapolate":
        train_k_range = tuple(args.train_k_set_range)
        test_k_range = (cell.test_k_set, cell.test_k_set)
    else:
        test_k_range = train_k_range

    label_fn = make_label_fn(cell.task, cell.k_value, args.d_max)
    train_iter = make_sample_iterator(train_rng, train_k_range, preset, label_fn)
    val_iter = make_sample_iterator(val_rng, train_k_range, preset, label_fn)
    test_iter = make_sample_iterator(test_rng, test_k_range, preset, label_fn)

    loss_fn = loss_for_task(cell.task)
    metric_name, metric_fn = metric_for_task(cell.task)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    history = {"step": [], "train_loss": [], "val": []}
    best_val = -float("inf") if cell.task != "distance" else float("inf")
    best_state = None

    t_start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        samples = [next(train_iter) for _ in range(args.batch_size)]
        bb = collate_to_batch(samples, in_dim, device)
        logits = model(bb.x, bb.edge_index, bb.batch, bb.set_batch)
        loss = loss_fn(logits, bb.y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            ys, preds = [], []
            with torch.no_grad():
                for _ in range(args.val_batches):
                    samples_v = [next(val_iter) for _ in range(args.batch_size)]
                    vb = collate_to_batch(samples_v, in_dim, device)
                    out = model(vb.x, vb.edge_index, vb.batch, vb.set_batch)
                    if cell.task == "distance":
                        preds.append(out.cpu().numpy())
                    else:
                        preds.append(torch.sigmoid(out).cpu().numpy())
                    ys.append(vb.y.cpu().numpy())
            y_true = np.concatenate(ys)
            y_pred = np.concatenate(preds)
            val_score = metric_fn(y_true, y_pred)
            history["step"].append(step)
            history["train_loss"].append(float(loss.item()))
            history["val"].append(val_score)
            improve = (
                (val_score > best_val)
                if cell.task != "distance"
                else (val_score < best_val)
            )
            if not math.isnan(val_score) and improve:
                best_val = val_score
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for _ in range(args.test_batches):
            samples_t = [next(test_iter) for _ in range(args.batch_size)]
            tb = collate_to_batch(samples_t, in_dim, device)
            out = model(tb.x, tb.edge_index, tb.batch, tb.set_batch)
            if cell.task == "distance":
                preds.append(out.cpu().numpy())
            else:
                preds.append(torch.sigmoid(out).cpu().numpy())
            ys.append(tb.y.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    test_score = metric_fn(y_true, y_pred)
    elapsed = time.time() - t_start
    return test_score, elapsed, history


# ============================================================================
# Statistics
# ============================================================================


def t_ci_95(values: Sequence[float]) -> Tuple[float, float]:
    """95% Student-t confidence interval. Returns (mean, half-width)."""
    a = np.array([v for v in values if not math.isnan(v)], dtype=np.float64)
    n = a.size
    if n < 2:
        return (float(a.mean()) if n else float("nan"), float("nan"))
    mean = a.mean()
    se = a.std(ddof=1) / math.sqrt(n)
    # Two-sided t critical for n-1 dof at 0.95
    from scipy.stats import t as t_dist

    crit = float(t_dist.ppf(0.975, df=n - 1))
    return float(mean), float(crit * se)


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect size for paired samples. d_z = mean(diff) / std(diff)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    if diff.size < 2 or diff.std(ddof=1) == 0:
        return float("nan")
    return float(diff.mean() / diff.std(ddof=1))


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon. Returns p-value, or NaN if ties everywhere."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Drop NaNs pairwise
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 5:
        return float("nan")
    if np.all(a == b):
        return 1.0
    from scipy.stats import wilcoxon

    try:
        stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat.pvalue)
    except ValueError:
        return float("nan")


def holm_bonferroni(
    pvalues: Dict[str, float], alpha: float = 0.05
) -> Dict[str, Tuple[float, bool]]:
    """Holm-Bonferroni correction. Returns {key: (adj_p, reject)}."""
    items = [(k, p) for k, p in pvalues.items() if not math.isnan(p)]
    items.sort(key=lambda kp: kp[1])
    m = len(items)
    out = {}
    prev_adj = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        adj = max(adj, prev_adj)  # enforce monotonicity
        prev_adj = adj
        out[k] = (adj, adj < alpha)
    for k, p in pvalues.items():
        if math.isnan(p):
            out[k] = (float("nan"), False)
    return out


# ============================================================================
# Output formatters
# ============================================================================


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def _fmt_score(mean: float, half: float, higher_is_better: bool, best: float) -> str:
    if math.isnan(mean):
        return "n/a"
    bold = abs(mean - best) < 1e-9
    s = f"{mean:.3f} \u00b1 {half:.3f}"
    if bold:
        s = f"**{s}**"
    return s


def write_markdown(path: Path, summary, args, equal_info):
    out = ["# Synthetic-benchmark results", ""]
    out += [
        f"- Run: `{args.out_dir}`",
        f"- Seeds: {args.seeds}",
        f"- Steps per seed: {args.steps}",
        f"- Equalization mode: `{args.equalize_params}`",
        "",
    ]
    if equal_info:
        out += [
            "## Parameter budgets",
            "",
            "| model | hidden | parameters |",
            "|---|---:|---:|",
        ]
        hiddens, params = equal_info
        for k in MODEL_KEYS:
            out.append(f"| {MODEL_KEYS[k]} | {hiddens[k]} | {params[k]:,} |")
        out.append("")

    # One block per task
    by_task = defaultdict(list)
    for cell_name, cell_data in summary.items():
        by_task[cell_data["task"]].append((cell_name, cell_data))

    for task, cells in by_task.items():
        metric = cells[0][1]["metric"]
        higher = metric == "AP"
        out += [
            f"## Task: `{task}`  (metric = {metric}, "
            f"{'higher' if higher else 'lower'} is better)",
            "",
        ]
        header = (
            "| K / setting | " + " | ".join(MODEL_KEYS[k] for k in MODEL_KEYS) + " |"
        )
        sep = "|---|" + "---|" * len(MODEL_KEYS)
        out += [header, sep]
        for cell_name, cd in cells:
            row_label = cell_name.replace(f"{task}_", "")
            cell_means = {k: cd["per_model"][k]["mean"] for k in MODEL_KEYS}
            best = (max if higher else min)(
                v for v in cell_means.values() if not math.isnan(v)
            )
            cells_fmt = []
            for k in MODEL_KEYS:
                pm = cd["per_model"][k]
                cells_fmt.append(_fmt_score(pm["mean"], pm["ci95_half"], higher, best))
            out.append(f"| {row_label} | " + " | ".join(cells_fmt) + " |")
        out.append("")

    out += [
        "## Pairwise Wilcoxon signed-rank tests (Holm-Bonferroni adjusted within cell)",
        "",
    ]
    out += [
        "| cell | comparison | p (adj) | Cohen's d_z | reject H0 |",
        "|---|---|---:|---:|:---:|",
    ]
    for cell_name, cd in summary.items():
        for pair_name, pdat in cd.get("pairwise", {}).items():
            mark = "yes" if pdat["reject"] else "no"
            p_str = "n/a" if math.isnan(pdat["p_adj"]) else f"{pdat['p_adj']:.3g}"
            d_str = "n/a" if math.isnan(pdat["d"]) else f"{pdat['d']:+.2f}"
            out.append(f"| {cell_name} | {pair_name} | {p_str} | {d_str} | {mark} |")

    path.write_text("\n".join(out) + "\n")


def _latex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("_", r"\_").replace("#", r"\#")


def write_latex(path: Path, summary, args, equal_info):
    """Booktabs results table, one block per task. Bolds the best per row."""
    lines = []
    lines += [
        r"% Auto-generated by synth_benchmark.py",
        r"% Mean +/- 95\% Student-t CI half-width over " + str(args.seeds) + " seeds.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Synthetic-benchmark results.\ "
        r"GraphSetConv variants vs.\ GCN+DeepSets and GCN+SetTransformer "
        r"pipelines.\ Mean $\pm$ 95\% CI over "
        + str(args.seeds)
        + r" seeds; \textbf{bold} = best in row.}",
        r"\label{tab:synth_main}",
        r"\begin{tabular}{l" + "c" * len(MODEL_KEYS) + r"}",
        r"\toprule",
    ]
    cols = " & ".join(_latex_escape(MODEL_KEYS[k]) for k in MODEL_KEYS)
    lines.append("Setting & " + cols + r" \\")
    lines.append(r"\midrule")

    by_task = defaultdict(list)
    for cell_name, cell_data in summary.items():
        by_task[cell_data["task"]].append((cell_name, cell_data))

    for task, cells in by_task.items():
        metric = cells[0][1]["metric"]
        arrow = r"$\uparrow$" if metric == "AP" else r"$\downarrow$"
        lines.append(
            r"\multicolumn{%d}{l}{\textit{Task: %s\ (%s\ %s)}} \\"
            % (1 + len(MODEL_KEYS), _latex_escape(task), metric, arrow)
        )
        higher = metric == "AP"
        for cell_name, cd in cells:
            row_label = cell_name.replace(f"{task}_", "")
            cell_means = {k: cd["per_model"][k]["mean"] for k in MODEL_KEYS}
            best = (max if higher else min)(
                v for v in cell_means.values() if not math.isnan(v)
            )
            cells_fmt = []
            for k in MODEL_KEYS:
                pm = cd["per_model"][k]
                m, h = pm["mean"], pm["ci95_half"]
                if math.isnan(m):
                    cells_fmt.append("n/a")
                else:
                    s = f"{m:.3f}\\,$\\pm$\\,{h:.3f}"
                    if abs(m - best) < 1e-9:
                        s = r"\textbf{" + s + "}"
                    cells_fmt.append(s)
            lines.append(
                _latex_escape(row_label) + " & " + " & ".join(cells_fmt) + r" \\"
            )
        lines.append(r"\addlinespace")

    if equal_info:
        hiddens, params = equal_info
        lines.append(r"\midrule")
        param_cells = " & ".join(f"{params[k]:,}" for k in MODEL_KEYS)
        lines.append(r"\textit{Parameters} & " + param_cells + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_pairwise_latex(path: Path, summary, args):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Paired Wilcoxon signed-rank tests over "
        + str(args.seeds)
        + r" seeds, Holm--Bonferroni adjusted within cell.\ "
        r"$d_z$ is Cohen's effect size for paired samples.}",
        r"\label{tab:synth_pairwise}",
        r"\begin{tabular}{llrrc}",
        r"\toprule",
        r"Cell & Comparison & $p_{\mathrm{adj}}$ & $d_z$ & "
        r"reject $H_0$ \\",
        r"\midrule",
    ]
    for cell_name, cd in summary.items():
        if not cd.get("pairwise"):
            continue
        first = True
        for pair_name, pdat in cd["pairwise"].items():
            cell_text = _latex_escape(cell_name) if first else ""
            first = False
            p = pdat["p_adj"]
            d = pdat["d"]
            mark = r"\checkmark" if pdat["reject"] else "--"
            p_str = "n/a" if math.isnan(p) else f"{p:.3g}"
            d_str = "n/a" if math.isnan(d) else f"{d:+.2f}"
            lines.append(
                f"{cell_text} & {_latex_escape(pair_name)} & "
                f"{p_str} & {d_str} & {mark} " + r"\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


# ============================================================================
# Cell construction
# ============================================================================


def build_cells(args) -> List[CellSpec]:
    cells: List[CellSpec] = []
    tasks = (
        args.tasks
        if args.tasks != ["all"]
        else ["ring", "ball", "distance", "ring_extrapolate"]
    )
    if "ring" in tasks:
        for k in args.ring_k_list:
            cells.append(CellSpec(task="ring", k_value=k))
    if "ball" in tasks:
        for k in args.ball_k_list:
            cells.append(CellSpec(task="ball", k_value=k))
    if "distance" in tasks:
        cells.append(
            CellSpec(task="distance", k_value=args.d_max, extra={"d_max": args.d_max})
        )
    if "ring_extrapolate" in tasks:
        for ts in args.extrapolate_test_set_sizes:
            cells.append(
                CellSpec(
                    task="ring_extrapolate",
                    k_value=args.extrapolate_k_value,
                    test_k_set=ts,
                )
            )
    return cells


# ============================================================================
# Main runner
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    # Compute / global
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/synth_run")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 1 seed, few steps, two K values.",
    )

    # Model architecture
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--test-batches", type=int, default=32)
    p.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds per cell (publication: >=10).",
    )
    p.add_argument("--seed-offset", type=int, default=0)

    # Tasks
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "ring", "ball", "distance", "ring_extrapolate"],
    )
    p.add_argument("--ring-k-list", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--ball-k-list", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument(
        "--d-max", type=int, default=5, help="Cap for distance regression target."
    )
    p.add_argument(
        "--extrapolate-k-value",
        type=int,
        default=2,
        help="Underlying ring K for set-size extrapolation task.",
    )
    p.add_argument(
        "--extrapolate-test-set-sizes", nargs="+", type=int, default=[8, 10, 12]
    )
    p.add_argument("--train-k-set-range", nargs=2, type=int, default=[3, 5])

    # Synthetic data
    p.add_argument("--graph-preset", choices=list(PRESETS.keys()), default="default")
    p.add_argument(
        "--k-set-range",
        nargs=2,
        type=int,
        default=[3, 7],
        help="Set size range used for everything except extrapolate.",
    )

    # Equalization & model selection
    p.add_argument(
        "--equalize-params", choices=["none", "smaller", "larger"], default="none"
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["gsc-ca", "gsc-bc", "gcn-ds", "gcn-st"],
        choices=list(MODEL_KEYS.keys()),
        help="Subset of model keys to run. Default = the v1/v2 "
        "comparison set; add gsc-v3-ca, gsc-v3-bc, gsc-v3-r "
        "to include the v3 block.",
    )
    p.add_argument(
        "--num-pool-tokens",
        type=int,
        default=4,
        help="K for v3 multi-token pool. Ignored by non-v3 models.",
    )
    p.add_argument(
        "--num-registers",
        type=int,
        default=2,
        help="Register tokens for gsc-v3-r. Ignored by other models.",
    )

    args = p.parse_args()

    if args.quick:
        args.steps = 1200
        args.seeds = 1
        args.eval_every = 150
        args.val_batches = 2
        args.test_batches = 4
        args.ring_k_list = [1, 3]
        args.ball_k_list = [1]
        args.extrapolate_test_set_sizes = [8]
    return args


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = PRESETS[args.graph_preset]
    in_dim = 4
    device = args.device

    print(f"=== synth_benchmark.py ===")
    print(f"out-dir = {out_dir}")
    print(f"device  = {device}")
    print(
        f"preset  = {preset.name} (N in [{preset.n_min},{preset.n_max}], "
        f"p in [{preset.p_min:.2f},{preset.p_max:.2f}])"
    )
    print(f"models  = {[MODEL_KEYS[k] for k in args.models]}")
    print(f"seeds   = {args.seeds}")
    print(f"steps   = {args.steps}")

    # Resolve param-equalized hidden dims (for the full registry, even if
    # only some models are run, so the report is internally consistent).
    hiddens, params = equalize_models(
        in_dim,
        args.hidden,
        args.layers,
        args.num_heads,
        "ring",
        args.equalize_params,
        num_pool_tokens=args.num_pool_tokens,
        num_registers=args.num_registers,
    )
    print(f"\n[equalize-params={args.equalize_params}]")
    for k in MODEL_KEYS:
        print(f"  {MODEL_KEYS[k]:<28} hidden={hiddens[k]:>4}  params={params[k]:>8,}")

    cells = build_cells(args)
    print(f"\n{len(cells)} cells:")
    for c in cells:
        print(f"  - {c.name}")

    # Quick label-density audit so the user sees the difficulty of each cell
    # before training kicks off.
    print(f"\n--- Label-density audit ({preset.name} graphs, n=400 samples) ---")
    for c in cells:
        label_fn = make_label_fn(c.task, c.k_value, args.d_max)
        kr = (
            (c.test_k_set, c.test_k_set)
            if c.task == "ring_extrapolate"
            else tuple(args.k_set_range)
        )
        a, b = audit_label_density(0, kr, preset, label_fn, n=400)
        if c.task == "distance":
            print(f"  {c.name:<32}  target mean={a:.2f}  std={b:.2f}")
        else:
            print(f"  {c.name:<32}  pos density={a:.4f}  frac graphs w/ anchor={b:.3f}")

    # Persist run config
    config_payload = {
        "args": vars(args),
        "model_param_counts": params,
        "model_hidden_dims": hiddens,
        "preset": asdict(preset),
        "cells": [asdict(c) for c in cells],
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }
    write_json(out_dir / "config.json", config_payload)

    # Main loop
    raw_results = {}  # cell_name -> {model_key -> [scores]}
    timings = {}  # cell_name -> {model_key -> [seconds]}
    history_records = {}  # cell_name -> {model_key -> {seed -> hist}}
    flat_csv_rows: List[Dict] = []

    for cell in cells:
        cell_name = cell.name
        raw_results[cell_name] = {k: [] for k in args.models}
        timings[cell_name] = {k: [] for k in args.models}
        history_records[cell_name] = {k: {} for k in args.models}
        print(f"\n========== cell: {cell_name} ==========")
        for mkey in args.models:
            display = MODEL_KEYS[mkey]
            for sidx in range(args.seeds):
                seed = args.seed_offset + sidx
                model = build_model(
                    mkey,
                    in_dim,
                    hiddens[mkey],
                    args.layers,
                    args.num_heads,
                    cell.task,
                    dropout=args.dropout,
                    num_pool_tokens=args.num_pool_tokens,
                    num_registers=args.num_registers,
                ).to(device)
                p_count = count_params(model)
                score, elapsed, hist = train_one(
                    model,
                    cell,
                    args,
                    seed,
                    in_dim,
                    preset,
                    device,
                )
                raw_results[cell_name][mkey].append(score)
                timings[cell_name][mkey].append(elapsed)
                history_records[cell_name][mkey][seed] = hist
                print(
                    f"  [{display:<24}] seed={seed:>2}  "
                    f"score={score:.4f}  ({elapsed:.1f}s, {p_count:,} params)"
                )
                flat_csv_rows.append(
                    {
                        "cell": cell_name,
                        "task": cell.task,
                        "k_value": cell.k_value,
                        "test_k_set": cell.test_k_set
                        if cell.test_k_set is not None
                        else "",
                        "model_key": mkey,
                        "model_name": display,
                        "seed": seed,
                        "hidden": hiddens[mkey],
                        "params": p_count,
                        "score": score,
                        "wall_seconds": round(elapsed, 2),
                        "metric": metric_for_task(cell.task)[0],
                    }
                )
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    # Aggregate
    summary = {}
    for cell in cells:
        cell_name = cell.name
        metric_name = metric_for_task(cell.task)[0]
        per_model = {}
        for mkey in MODEL_KEYS:
            scores = raw_results.get(cell_name, {}).get(mkey, [])
            if not scores:
                per_model[mkey] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "ci95_half": float("nan"),
                    "n": 0,
                    "scores": [],
                    "params": params[mkey],
                    "hidden": hiddens[mkey],
                }
                continue
            mean, half = t_ci_95(scores)
            per_model[mkey] = {
                "mean": mean,
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "ci95_half": half,
                "n": len(scores),
                "scores": [float(s) for s in scores],
                "params": params[mkey],
                "hidden": hiddens[mkey],
            }

        # Pairwise (only over models that were actually run with >=2 seeds)
        active = [k for k in args.models if per_model[k]["n"] >= 2]
        pairwise_raw = {}
        for k1, k2 in combinations(active, 2):
            a = per_model[k1]["scores"]
            b = per_model[k2]["scores"]
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            p = wilcoxon_signed_rank(a, b)
            d = cohens_d_paired(a, b)
            pair_label = f"{MODEL_KEYS[k1]} vs {MODEL_KEYS[k2]}"
            pairwise_raw[pair_label] = (p, d)

        adj = holm_bonferroni({k: v[0] for k, v in pairwise_raw.items()})
        pairwise = {}
        for label, (p_raw, d) in pairwise_raw.items():
            adj_p, reject = adj[label]
            pairwise[label] = {
                "p_raw": p_raw,
                "p_adj": adj_p,
                "reject": bool(reject),
                "d": d,
            }

        summary[cell_name] = {
            "task": cell.task,
            "k_value": cell.k_value,
            "test_k_set": cell.test_k_set,
            "metric": metric_name,
            "per_model": per_model,
            "pairwise": pairwise,
            "wall_seconds": {
                k: float(np.mean(v)) if v else float("nan")
                for k, v in timings.get(cell_name, {}).items()
            },
        }

    # Write outputs
    write_json(
        out_dir / "raw.json",
        {
            "summary": summary,
            "raw_scores": raw_results,
            "timings": timings,
            "history": history_records,
        },
    )
    write_csv(out_dir / "summary.csv", flat_csv_rows)
    write_markdown(out_dir / "summary.md", summary, args, (hiddens, params))
    write_latex(out_dir / "summary.tex", summary, args, (hiddens, params))
    write_pairwise_latex(out_dir / "pairwise.tex", summary, args)

    # Console summary
    print(f"\n========== summary ==========\n")
    for cell in cells:
        cn = cell.name
        cd = summary[cn]
        metric = cd["metric"]
        print(f"[{cn}]  metric={metric}")
        for mkey in args.models:
            pm = cd["per_model"][mkey]
            if pm["n"] == 0:
                continue
            print(
                f"  {MODEL_KEYS[mkey]:<28} "
                f"{pm['mean']:.4f} \u00b1 {pm['ci95_half']:.4f}  "
                f"(n={pm['n']}, params={pm['params']:,})"
            )
        if cd["pairwise"]:
            print("  pairwise (Holm-adj):")
            for label, dat in cd["pairwise"].items():
                tag = "*" if dat["reject"] else " "
                p_str = "n/a" if math.isnan(dat["p_adj"]) else f"{dat['p_adj']:.3g}"
                d_str = "n/a" if math.isnan(dat["d"]) else f"{dat['d']:+.2f}"
                print(f"   {tag} {label}:  p_adj={p_str},  d_z={d_str}")
        print()

    print(f"All outputs written to: {out_dir}")
    print(f"  raw.json           full per-seed scores + history")
    print(f"  summary.csv        flat table for re-analysis")
    print(f"  summary.md         human-readable summary")
    print(f"  summary.tex        LaTeX main results table")
    print(f"  pairwise.tex       LaTeX pairwise significance table")
    print(f"  config.json        run configuration and environment")


if __name__ == "__main__":
    run(parse_args())
