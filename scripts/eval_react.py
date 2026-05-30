"""
bh_benchmark.py
===============

Buchwald-Hartwig yield prediction benchmark for GraphSetConv vs. pipeline
baselines (GCN + DeepSets / SetTransformer). 2D molecular graphs only — no
3D coordinates needed for this dataset.

Each reaction is a SET of 4 small molecular graphs:
    {aryl halide, ligand, base, additive}
The architectural test: does interleaved set-graph reasoning help on a
real-data task where the 4 elements are categorically distinct?

Two evaluation protocols
------------------------
1. Random 70/10/20 split (--protocol random, default). 5+ seeds.
   Comparable to Doyle 2018 RF (R^2=0.92), Yield-BERT (R^2=0.94),
   AGNN/MPNN papers. Headline result.

2. Sandfort/Schwaller out-of-sample splits (--protocol test1..test4).
   Each TestN holds out a different subset of additives. Tests
   extrapolation to unseen functional groups. Standard reference: R^2~0.7
   for Yield-BERT.

Models compared
---------------
gsc-v1-ca   GraphSetConv (broadcast set context, original)
gsc-v2-ca   GraphSetConv (per-node cross-attn, single token)
gcn-ds      GCN encoder + DeepSets head
gcn-st      GCN encoder + SetTransformer head

All variants share an encoder across the 4 reactant types — the
architectural claim is about interleaved set reasoning, not slot
specialization.

Performance notes (vs. the previous version)
--------------------------------------------
The old script was CPU-bound on a 5090 due to (a) per-graph tensor
allocation in collate, (b) implicit `.max().item()` syncs inside the
pooling helpers (called twice per forward), (c) per-parameter `isfinite`
checks in the training loop, and (d) full validation on every epoch.
The fixed version:
  - Caches per-reaction packed tensors on first access; collat--exclude={'.git', '.venv'}e becomes
    O(reactions-in-batch) torch.cats instead of O(graphs-in-batch).
  - Threads num_groups through pool functions and to_dense_batch.
  - Drops the per-parameter gradient `isfinite` loop; relies on
    grad-clip + a single `loss.isfinite()` check.
  - Adds --val-every (default 3) and --eval-batch-size (default 128).
  - Patience semantics unchanged: still "epochs since improvement".

Param-equalization (--equalize-params {none, smaller, larger}) and the
training-set-size ablation behave exactly as before.

Outputs (per --out-dir/<protocol>/):
    raw.json       per-seed scores (R^2, RMSE, MAE), params, timings
    summary.csv    flat re-analyzable table
    summary.md     human-readable report
    summary.tex    booktabs LaTeX
    pairwise.tex   booktabs Wilcoxon table
    config.json    exact run config + env

Usage
-----
    # Smoke test
    python bh_benchmark.py --quick

    # Headline: random 70/10/20, all 5 models, 5 seeds, natural budgets
    python bh_benchmark.py --protocol random --seeds 5 \\
        --out-dir results/bh_random

    # Reproduce previous-version timing (slow): pass --val-every 1
    python bh_benchmark.py --protocol random --seeds 5 --val-every 1 \\
        --out-dir results/bh_random_oldspeed
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from react_data import (
    ATOM_FEAT_DIM,
    BHReaction,
    Mol2DGraph,
    TargetStats,
    compute_target_stats,
    get_atom_feat_dim,
    load_bh_full,
    load_bh_split,
    random_split,
    val_split_from_train,
    val_split_stratified_by_additive,
)

# Reuse blocks from existing modules.
from graph_set_transformer.models.gst_v2 import GraphSetConv


# =============================================================================
# Pure-torch helpers (no torch_scatter dependency)
# =============================================================================
# IMPORTANT: num_groups is now a required argument. Inferring it from the batch
# tensor requires `batch.max().item()`, which forces a CUDA sync on every
# forward — extremely expensive on a fast GPU when graphs are small. Callers
# always know num_groups statically (we computed it in collate), so we pass it.


def global_add_pool_safe(
    x: torch.Tensor, batch: torch.Tensor, num_groups: int
) -> torch.Tensor:
    out = x.new_zeros(num_groups, x.size(-1))
    out.index_add_(0, batch, x)
    return out


def global_mean_pool_safe(
    x: torch.Tensor, batch: torch.Tensor, num_groups: int
) -> torch.Tensor:
    out = x.new_zeros(num_groups, x.size(-1))
    out.index_add_(0, batch, x)
    counts = x.new_zeros(num_groups)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(-1)


# =============================================================================
# Set heads for the pipeline baselines
# =============================================================================


class DeepSetsHead(nn.Module):
    """phi(z) -> sum -> rho(z); the canonical permutation-invariant set fn."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        z_graph: torch.Tensor,
        set_batch: torch.Tensor,
        num_sets: int,
    ) -> torch.Tensor:
        h = self.phi(z_graph)
        z_set = global_add_pool_safe(h, set_batch, num_sets)
        return self.rho(z_set)


class SetTransformerHead(nn.Module):
    """Self-attention over per-graph embeddings + PMA seed pooling."""

    def __init__(
        self, dim: int, num_heads: int = 4, dropout: float = 0.1, ffn_mult: int = 2
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.ln3 = nn.LayerNorm(dim)
        self.pma = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        z_graph: torch.Tensor,
        set_batch: torch.Tensor,
        num_sets: int,
    ) -> torch.Tensor:
        from torch_geometric.utils import to_dense_batch

        # Passing batch_size avoids the implicit `.max()` on set_batch.
        z_dense, mask = to_dense_batch(z_graph, set_batch, batch_size=num_sets)
        zn = self.ln1(z_dense)
        attn, _ = self.mha(zn, zn, zn, key_padding_mask=~mask, need_weights=False)
        z_dense = z_dense + attn
        z_dense = z_dense + self.ffn(self.ln2(z_dense))
        S = z_dense.size(0)
        seed = self.seed.expand(S, -1, -1)
        zn = self.ln3(z_dense)
        z_set, _ = self.pma(seed, zn, zn, key_padding_mask=~mask, need_weights=False)
        return z_set.squeeze(1)


# =============================================================================
# Model registry and build
# =============================================================================


MODEL_KEYS = {
    "gsc-v1-ca": "GraphSetConv-V1",
    "gsc-v2-ca": "GraphSetConv-V2-CrossAttn",
    "gcn-ds": "GCN+DeepSets",
    "gcn-st": "GCN+SetTransformer",
}

PIPELINE_KEYS = {"gcn-ds", "gcn-st"}
GSC_KEYS = {"gsc-v1-ca", "gsc-v2-ca"}


class GCNStack(nn.Module):
    """Plain GCN trunk used by the pipeline baselines."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        d_prev = in_dim
        for _ in range(num_layers):
            self.layers.append(GCNConv(d_prev, hidden, improved=True))
            self.norms.append(nn.LayerNorm(hidden))
            self.dropouts.append(nn.Dropout(dropout))
            d_prev = hidden

    def forward(self, x, edge_index):
        for gcn, ln, dp in zip(self.layers, self.norms, self.dropouts):
            x = gcn(x, edge_index)
            x = F.silu(ln(x))
            x = dp(x)
        return x


class BHModel(nn.Module):
    """Single shared-encoder pipeline. Atom features run through a small
    input projection -> GCN/GraphSetConv stack -> per-graph pool -> per-set
    aggregation (DeepSets/SetTransformer for baselines, or already done
    inside the GraphSetConv blocks for v1/v2) -> regression head.
    """

    def __init__(
        self,
        model_key: str,
        num_layers: int,
        hidden: int,
        num_heads: int,
        dropout: float,
        atom_feat_dim: int = ATOM_FEAT_DIM,
    ):
        super().__init__()
        self.model_key = model_key
        self.input_proj = nn.Linear(atom_feat_dim, hidden)

        if model_key in PIPELINE_KEYS:
            self.is_pipeline = True
            self.encoder = GCNStack(hidden, hidden, num_layers, dropout=dropout)
            if model_key == "gcn-ds":
                self.set_head = DeepSetsHead(hidden, dropout=dropout)
            else:
                self.set_head = SetTransformerHead(hidden, num_heads, dropout=dropout)
        else:
            self.is_pipeline = False
            blocks = []
            for _ in range(num_layers):
                if model_key == "gsc-v1-ca":
                    blocks.append(
                        GraphSetConv(
                            filters=hidden,
                            in_channels=hidden,
                            num_heads=num_heads,
                            mhsa_dropout=dropout,
                            ffn_dropout=dropout,
                            node_set_mode="broadcast",
                        )
                    )
                elif model_key == "gsc-v2-ca":
                    blocks.append(
                        GraphSetConv(
                            filters=hidden,
                            in_channels=hidden,
                            num_heads=num_heads,
                            mhsa_dropout=dropout,
                            ffn_dropout=dropout,
                            node_set_mode="cross_attn",
                        )
                    )
                else:
                    raise ValueError(f"unknown gsc variant: {model_key}")
            self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch_obj):
        x = self.input_proj(batch_obj.x)
        if self.is_pipeline:
            x = self.encoder(x, batch_obj.edge_index)
            z_graph = global_mean_pool_safe(x, batch_obj.batch, batch_obj.n_graphs)
            z_set = self.set_head(z_graph, batch_obj.set_batch, batch_obj.n_sets)
        else:
            for blk in self.blocks:
                x = blk(x, batch_obj.edge_index, batch_obj.batch, batch_obj.set_batch)
            z_graph = global_mean_pool_safe(x, batch_obj.batch, batch_obj.n_graphs)
            z_set = global_mean_pool_safe(
                z_graph, batch_obj.set_batch, batch_obj.n_sets
            )
        return self.head(z_set).squeeze(-1)


def build_model(
    key: str,
    num_layers: int,
    hidden: int,
    num_heads: int,
    dropout: float,
    atom_feat_dim: int = ATOM_FEAT_DIM,
) -> nn.Module:
    return BHModel(
        model_key=key,
        num_layers=num_layers,
        hidden=hidden,
        num_heads=num_heads,
        dropout=dropout,
        atom_feat_dim=atom_feat_dim,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Param equalization (unchanged)
# =============================================================================


def find_hidden_for_target_params(
    builder: Callable[[int], nn.Module],
    target: int,
    num_heads: int,
    lo: int = 16,
    hi: int = 1024,
) -> Tuple[int, int]:
    best_h, best_p, best_err = lo, None, float("inf")
    for h in range(lo, hi + 1):
        if h % num_heads != 0:
            continue
        try:
            m = builder(h)
            p = count_params(m)
        except Exception:
            continue
        finally:
            try:
                del m
            except NameError:
                pass
        err = abs(p - target)
        if err < best_err:
            best_err, best_h, best_p = err, h, p
        if best_p is not None and p > target * 4:
            break
    return best_h, (best_p if best_p is not None else 0)


def equalize_models(args) -> Tuple[Dict[str, int], Dict[str, int]]:
    base_hidden = args.hidden
    atom_feat_dim = get_atom_feat_dim(args.feature_mode)

    def build_for(key: str, hidden: int) -> nn.Module:
        return build_model(
            key,
            args.layers,
            hidden,
            args.num_heads,
            args.dropout,
            atom_feat_dim=atom_feat_dim,
        )

    base_hiddens = {k: base_hidden for k in args.models}
    base_params: Dict[str, int] = {}
    for k in args.models:
        m = build_for(k, base_hidden)
        base_params[k] = count_params(m)
        del m

    if args.equalize_params == "none":
        return base_hiddens, base_params

    if args.equalize_params == "smaller":
        active_pipelines = [k for k in PIPELINE_KEYS if k in args.models]
        if not active_pipelines:
            raise ValueError(
                "--equalize-params smaller requires gcn-ds or gcn-st in --models."
            )
        target_p = min(base_params[k] for k in active_pipelines)
        hiddens = dict(base_hiddens)
        params = dict(base_params)
        for k in args.models:
            if k in PIPELINE_KEYS:
                continue
            h, p = find_hidden_for_target_params(
                lambda hidden, kk=k: build_for(kk, hidden),
                target_p,
                args.num_heads,
            )
            hiddens[k] = h
            params[k] = p
        return hiddens, params

    if args.equalize_params == "larger":
        active_gsc = [k for k in GSC_KEYS if k in args.models]
        if not active_gsc:
            raise ValueError(
                "--equalize-params larger requires gsc-v1-ca or gsc-v2-ca in --models."
            )
        target_p = max(base_params[k] for k in active_gsc)
        hiddens = dict(base_hiddens)
        params = dict(base_params)
        for k in args.models:
            if k in GSC_KEYS:
                continue
            h, p = find_hidden_for_target_params(
                lambda hidden, kk=k: build_for(kk, hidden),
                target_p,
                args.num_heads,
            )
            hiddens[k] = h
            params[k] = p
        return hiddens, params

    raise ValueError(f"Unknown --equalize-params: {args.equalize_params!r}")


# =============================================================================
# Per-reaction tensor packing (cached on first access)
# =============================================================================
# Each BHReaction has 4 small graphs (aryl halide / ligand / base / additive).
# In the old version, collate() rebuilt each batch by looping over every
# graph of every reaction and torch.cat-ing tiny tensors -- 32*4 = 128 cats
# per minibatch. Now we pack each reaction's 4 graphs into single tensors
# *once*, and collate just offsets and concatenates per-reaction packs.
#
# We use a side-cache keyed by id(reaction) instead of mutating BHReaction,
# so the data-loading module doesn't need to know about packing.

_PACKED_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = {}


def _packed(rxn: BHReaction):
    """Return the packed form of `rxn`: (x, edge_index, batch_local, n_atoms, n_graphs).

    `batch_local` runs from 0..n_graphs-1, indicating which graph each atom
    belongs to within this reaction. The caller offsets these with the
    running graph counter when assembling a batch.
    """
    key = id(rxn)
    cached = _PACKED_CACHE.get(key)
    if cached is not None:
        return cached

    xs: List[torch.Tensor] = []
    edge_chunks: List[torch.Tensor] = []
    batch_chunks: List[torch.Tensor] = []
    n_off = 0
    for gi, g in enumerate(rxn.graphs):
        xs.append(g.x)
        if g.edge_index.numel() > 0:
            edge_chunks.append(g.edge_index + n_off)
        batch_chunks.append(torch.full((g.num_nodes,), gi, dtype=torch.long))
        n_off += g.num_nodes

    packed = (
        torch.cat(xs, dim=0).contiguous(),
        torch.cat(edge_chunks, dim=1).contiguous()
        if edge_chunks
        else torch.zeros((2, 0), dtype=torch.long),
        torch.cat(batch_chunks, dim=0).contiguous(),
        n_off,
        len(rxn.graphs),
    )
    _PACKED_CACHE[key] = packed
    return packed


def prewarm_pack_cache(*items_lists: List[BHReaction]) -> None:
    """Pre-pack every reaction in the given lists. Avoids paying the packing
    cost on the first batch of training (where it would otherwise look like
    a slow first epoch)."""
    for items in items_lists:
        for rxn in items:
            _packed(rxn)


# =============================================================================
# Batch packing: each reaction = N (typically 4) small graphs
# =============================================================================


class BHPackedBatch:
    """Holds one mini-batch of BH reactions packed into flat tensors.

    Conventions:
        x:           [N_total_atoms, ATOM_FEAT_DIM]
        edge_index:  [2, E_total]
        batch:       [N_total_atoms]   atom -> graph_id (0..n_graphs-1)
        set_batch:   [n_graphs]        graph_id -> set_id (0..n_sets-1)
        y:           [n_sets]          yields, one per reaction
        n_graphs:    int               total graphs in this batch
        n_sets:      int               total reactions in this batch
    """

    __slots__ = ("x", "edge_index", "batch", "set_batch", "y", "n_graphs", "n_sets")

    def __init__(self, x, edge_index, batch, set_batch, y, n_graphs, n_sets):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.set_batch = set_batch
        self.y = y
        self.n_graphs = n_graphs
        self.n_sets = n_sets

    def to(self, device):
        # non_blocking only matters if the source tensors are pinned; harmless
        # when they're not.
        for attr in ("x", "edge_index", "batch", "set_batch", "y"):
            setattr(self, attr, getattr(self, attr).to(device, non_blocking=True))
        return self


def collate(samples: List[BHReaction]) -> BHPackedBatch:
    xs: List[torch.Tensor] = []
    edge_chunks: List[torch.Tensor] = []
    batch_chunks: List[torch.Tensor] = []
    set_chunks: List[torch.Tensor] = []
    ys: List[float] = []
    n_off = 0
    g_off = 0
    for sid, rxn in enumerate(samples):
        x_r, ei_r, batch_r, n_atoms_r, n_graphs_r = _packed(rxn)
        xs.append(x_r)
        if ei_r.numel() > 0:
            edge_chunks.append(ei_r + n_off)
        batch_chunks.append(batch_r + g_off)
        set_chunks.append(torch.full((n_graphs_r,), sid, dtype=torch.long))
        n_off += n_atoms_r
        g_off += n_graphs_r
        ys.append(rxn.y)

    edge_index = (
        torch.cat(edge_chunks, dim=1)
        if edge_chunks
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return BHPackedBatch(
        x=torch.cat(xs, dim=0),
        edge_index=edge_index,
        batch=torch.cat(batch_chunks, dim=0),
        set_batch=torch.cat(set_chunks, dim=0),
        y=torch.tensor(ys, dtype=torch.float32),
        n_graphs=g_off,
        n_sets=len(samples),
    )


# =============================================================================
# Statistics (Wilcoxon + Holm-Bonferroni; unchanged)
# =============================================================================


def t_ci_95(values):
    a = np.array([v for v in values if not math.isnan(v)], dtype=np.float64)
    n = a.size
    if n < 2:
        return (float(a.mean()) if n else float("nan"), float("nan"))
    from scipy.stats import t as t_dist

    se = a.std(ddof=1) / math.sqrt(n)
    return float(a.mean()), float(t_dist.ppf(0.975, df=n - 1) * se)


def cohens_d_paired(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    if diff.size < 2 or diff.std(ddof=1) == 0:
        return float("nan")
    return float(diff.mean() / diff.std(ddof=1))


def wilcoxon_signed_rank(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 5:
        return float("nan")
    if np.all(a == b):
        return 1.0
    from scipy.stats import wilcoxon

    try:
        return float(
            wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue
        )
    except ValueError:
        return float("nan")


def holm_bonferroni(pvalues, alpha=0.05):
    items = [(k, p) for k, p in pvalues.items() if not math.isnan(p)]
    items.sort(key=lambda kp: kp[1])
    m = len(items)
    out = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = max(min(1.0, p * (m - i)), prev)
        prev = adj
        out[k] = (adj, adj < alpha)
    for k, p in pvalues.items():
        if math.isnan(p):
            out[k] = (float("nan"), False)
    return out


# =============================================================================
# Train / eval
# =============================================================================


def iterate_minibatches(items, batch_size, rng, shuffle=True):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [items[j] for j in idx[i : i + batch_size]]


def eval_set(model, items, stats, args, device):
    """Compute RMSE, MAE, R^2 on the un-normalized (0..100) scale.

    Uses --eval-batch-size (default 128), independent of the train batch
    size, so val/test inference doesn't get throttled by the small train
    batches the model trains with.
    """
    if not items:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    model.eval()
    preds, truths = [], []
    eval_bs = max(1, args.eval_batch_size)
    with torch.no_grad():
        for samples in iterate_minibatches(
            items, eval_bs, random.Random(0), shuffle=False
        ):
            pb = collate(samples).to(device)
            pred_norm = model(pb)
            pred = stats.denormalize(pred_norm)
            preds.append(pred.cpu())
            truths.append(pb.y.cpu())
    if not preds:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    P = torch.cat(preds, dim=0)
    T = torch.cat(truths, dim=0)
    mae = (P - T).abs().mean().item()
    rmse = float(torch.sqrt(((P - T) ** 2).mean()))
    T_mean = T.mean()
    ss_res = ((T - P) ** 2).sum().item()
    ss_tot = ((T - T_mean) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_one(model, train, val, test, stats, args, seed, device):
    """Training loop.

    - val non-empty: evaluate every --val-every epochs; restore best-val state
      before test eval. Patience is in *epochs since improvement*, not val
      checks, so its meaning is independent of --val-every. Default
      --val-every=3 cuts validation cost ~3x without changing patience
      semantics.
    - val empty (--val-strategy none): trains for full --epochs, no early
      stopping; matches Schwaller's OOD protocol.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    use_val = len(val) > 0
    val_every = max(1, args.val_every)
    best_val = float("inf")
    best_state = None
    last_improve_epoch = 0
    history: List[Dict] = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        for samples in iterate_minibatches(train, args.batch_size, rng):
            pb = collate(samples).to(device)
            y_norm = stats.normalize(pb.y)
            pred = model(pb)
            loss = F.l1_loss(pred, y_norm)
            # Single CPU sync per step (loss.isfinite). The old per-parameter
            # gradient isfinite loop was the dominant overhead on a 5090; if a
            # NaN does sneak through, gradient clipping + the next step's
            # loss check catches it within an iteration.
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss.item() * pb.y.size(0)
            running_n += pb.y.size(0)

        if running_n == 0:
            print(f"    [seed={seed} epoch={epoch}] all batches non-finite; aborting")
            break

        # Validate every val_every epochs (and on the final epoch).
        do_val = use_val and (epoch % val_every == 0 or epoch == args.epochs)
        if do_val:
            v = eval_set(model, val, stats, args, device)
            history.append(
                {
                    "epoch": epoch,
                    "val_mae": v["mae"],
                    "val_rmse": v["rmse"],
                    "val_r2": v["r2"],
                }
            )
            if math.isfinite(v["mae"]) and v["mae"] < best_val:
                best_val = v["mae"]
                best_state = {
                    k: t.detach().clone() for k, t in model.state_dict().items()
                }
                last_improve_epoch = epoch
        elif not use_val:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": running_loss / max(running_n, 1),
                }
            )

        # Patience check (every epoch, regardless of whether we validated).
        if (
            use_val
            and last_improve_epoch > 0
            and (epoch - last_improve_epoch) >= args.patience
        ):
            break

    if use_val and best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_set(model, test, stats, args, device)
    if not math.isfinite(test_metrics["mae"]):
        for k in ("mae", "rmse", "r2"):
            test_metrics[k] = float("nan")
    return test_metrics, time.time() - t_start, history


# =============================================================================
# Output writers (unchanged)
# =============================================================================


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def write_markdown(path, summary, args, hiddens):
    out = [f"# BH yield benchmark", ""]
    out += [
        f"- Protocol: `{args.protocol}`",
        f"- Models: {len(summary['per_model'])}",
        f"- Seeds: {args.seeds}",
        f"- Epochs (max): {args.epochs}, patience: {args.patience}, val-every: {args.val_every}",
        f"- Equalization mode: `{args.equalize_params}`",
        f"- Feature mode: `{args.feature_mode}`",
        f"- Val strategy: `{args.val_strategy}`",
        f"- Hidden (default): {args.hidden}, Layers: {args.layers}",
        "",
    ]
    out += [
        "## Test metrics (mean ± 95% CI)",
        "",
        "| model | hidden | params | RMSE (%) | MAE (%) | R² |",
        "|---|---:|---:|---|---|---|",
    ]
    r2s = {k: pm["r2"]["mean"] for k, pm in summary["per_model"].items()}
    valid = [v for v in r2s.values() if not math.isnan(v)]
    best_r2 = max(valid) if valid else float("nan")
    for mkey, pm in summary["per_model"].items():
        rmse = pm["rmse"]
        mae = pm["mae"]
        r2 = pm["r2"]
        rmse_s = (
            f"{rmse['mean']:.3f} ± {rmse['ci95']:.3f}"
            if not math.isnan(rmse["mean"])
            else "n/a"
        )
        mae_s = (
            f"{mae['mean']:.3f} ± {mae['ci95']:.3f}"
            if not math.isnan(mae["mean"])
            else "n/a"
        )
        r2_s = (
            f"{r2['mean']:.3f} ± {r2['ci95']:.3f}"
            if not math.isnan(r2["mean"])
            else "n/a"
        )
        if not math.isnan(r2["mean"]) and abs(r2["mean"] - best_r2) < 1e-9:
            r2_s = f"**{r2_s}**"
        out.append(
            f"| {MODEL_KEYS[mkey]} | {hiddens[mkey]} | {pm['params']:,} | "
            f"{rmse_s} | {mae_s} | {r2_s} |"
        )
    out += [
        "",
        "## Pairwise Wilcoxon on RMSE (Holm-Bonferroni adjusted)",
        "",
        "| comparison | p (adj) | Cohen's d_z | reject H0 |",
        "|---|---:|---:|:---:|",
    ]
    for pair, pdat in summary.get("pairwise_rmse", {}).items():
        mark = "yes" if pdat["reject"] else "no"
        p_str = "n/a" if math.isnan(pdat["p_adj"]) else f"{pdat['p_adj']:.3g}"
        d_str = "n/a" if math.isnan(pdat["d"]) else f"{pdat['d']:+.2f}"
        out.append(f"| {pair} | {p_str} | {d_str} | {mark} |")
    path.write_text("\n".join(out) + "\n")


def _latex_escape(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("#", r"\#")


def write_latex(path, summary, args, hiddens):
    eq_note = ""
    if args.equalize_params != "none":
        eq_note = r"\ Param-equalized (\texttt{" + args.equalize_params + r"})."
    lines = [
        r"% Auto-generated by bh_benchmark.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Buchwald-Hartwig yield prediction ("
        + _latex_escape(args.protocol)
        + r" protocol; mean $\pm$ 95\% CI over "
        + str(args.seeds)
        + r" seeds; \textbf{bold} = best $R^2$)."
        + eq_note
        + r"}",
        r"\label{tab:bh_" + args.protocol + r"}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Hidden & Params & RMSE (\%) & MAE (\%) & $R^2$ \\",
        r"\midrule",
    ]
    r2s = {k: pm["r2"]["mean"] for k, pm in summary["per_model"].items()}
    valid = [v for v in r2s.values() if not math.isnan(v)]
    best_r2 = max(valid) if valid else float("nan")
    for mkey, pm in summary["per_model"].items():
        rmse, mae, r2 = pm["rmse"], pm["mae"], pm["r2"]
        rmse_s = (
            f"{rmse['mean']:.3f}\\,$\\pm$\\,{rmse['ci95']:.3f}"
            if not math.isnan(rmse["mean"])
            else "n/a"
        )
        mae_s = (
            f"{mae['mean']:.3f}\\,$\\pm$\\,{mae['ci95']:.3f}"
            if not math.isnan(mae["mean"])
            else "n/a"
        )
        r2_s = (
            f"{r2['mean']:.3f}\\,$\\pm$\\,{r2['ci95']:.3f}"
            if not math.isnan(r2["mean"])
            else "n/a"
        )
        if not math.isnan(r2["mean"]) and abs(r2["mean"] - best_r2) < 1e-9:
            r2_s = r"\textbf{" + r2_s + "}"
        lines.append(
            _latex_escape(MODEL_KEYS[mkey])
            + " & "
            + str(hiddens[mkey])
            + " & "
            + f"{pm['params']:,}"
            + " & "
            + rmse_s
            + " & "
            + mae_s
            + " & "
            + r2_s
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_pairwise_latex(path, summary, args):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{BH pairwise Wilcoxon signed-rank on RMSE, "
        + str(args.seeds)
        + r" seeds, Holm--Bonferroni adjusted.}",
        r"\label{tab:bh_" + args.protocol + r"_pw}",
        r"\begin{tabular}{lrrc}",
        r"\toprule",
        r"Comparison & $p_{\mathrm{adj}}$ & $d_z$ & reject $H_0$ \\",
        r"\midrule",
    ]
    if summary.get("pairwise_rmse"):
        for pair, pdat in summary["pairwise_rmse"].items():
            p, d = pdat["p_adj"], pdat["d"]
            mark = r"\checkmark" if pdat["reject"] else "--"
            p_str = "n/a" if math.isnan(p) else f"{p:.3g}"
            d_str = "n/a" if math.isnan(d) else f"{d:+.2f}"
            lines.append(
                _latex_escape(pair)
                + " & "
                + p_str
                + " & "
                + d_str
                + " & "
                + mark
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def train_eval_models(
    args,
    train: List[BHReaction],
    val: List[BHReaction],
    test: List[BHReaction],
    stats_device,
    hiddens: Dict[str, int],
    device: str,
    *,
    protocol_label: Optional[str] = None,
    extra_row_fields: Optional[Dict] = None,
    quiet: bool = False,
) -> Tuple[Dict, Dict, List[Dict], Dict]:
    raw_scores = {k: {"mae": [], "rmse": [], "r2": []} for k in args.models}
    timings = {k: [] for k in args.models}
    flat_rows: List[Dict] = []
    history: Dict[str, Dict[int, List]] = {k: {} for k in args.models}

    for mkey in args.models:
        display = MODEL_KEYS[mkey]
        if not quiet:
            print(f"\n--- {display} ---")
        for sidx in range(args.seeds):
            seed = args.seed_offset + sidx
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = build_model(
                mkey,
                args.layers,
                hiddens[mkey],
                args.num_heads,
                args.dropout,
                atom_feat_dim=get_atom_feat_dim(args.feature_mode),
            ).to(device)
            n_params = count_params(model)
            metrics, elapsed, hist = train_one(
                model,
                train,
                val,
                test,
                stats_device,
                args,
                seed,
                device,
            )
            if not quiet:
                print(
                    f"  seed={seed:>2}  test RMSE={metrics['rmse']:.3f}  "
                    f"MAE={metrics['mae']:.3f}  R2={metrics['r2']:.4f}  "
                    f"({elapsed:.1f}s, {n_params:,} params)"
                )
            for k in ("mae", "rmse", "r2"):
                raw_scores[mkey][k].append(metrics[k])
            timings[mkey].append(elapsed)
            history[mkey][seed] = hist
            row: Dict = {
                "protocol": protocol_label or args.protocol,
                "model_key": mkey,
                "model_name": display,
                "seed": seed,
                "hidden": hiddens[mkey],
                "params": n_params,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "wall_seconds": round(elapsed, 2),
            }
            if extra_row_fields:
                row.update(extra_row_fields)
            flat_rows.append(row)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    return raw_scores, timings, flat_rows, history


# =============================================================================
# Main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--data-root",
        default="data/bh",
        help="Will download Dreher_and_Doyle_input_data.xlsx here.",
    )
    p.add_argument("--out-dir", default="results/bh_run")
    p.add_argument("--quick", action="store_true")

    p.add_argument(
        "--protocol",
        choices=["random", "test1", "test2", "test3", "test4"]
        + [f"fullcv_{i:02d}" for i in range(1, 11)],
        default="random",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_KEYS.keys()),
        choices=list(MODEL_KEYS.keys()),
    )

    # Architecture
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-registers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--equalize-params",
        choices=["none", "smaller", "larger"],
        default="none",
    )

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Epochs since last val-improvement before early stop.",
    )
    p.add_argument(
        "--batch-size", type=int, default=32, help="Reactions per training batch."
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
        help="Reactions per val/test batch (inference only; can be "
        "much larger than --batch-size since no grads).",
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=3,
        help="Run validation every N epochs (default 3). Set to 1 "
        "for the previous-version behaviour. Patience is in "
        "epochs and unaffected by this flag.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)

    # Random split
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument(
        "--val-fraction-from-train",
        type=float,
        default=0.1,
    )

    # Atom features and val/early-stop strategy
    p.add_argument(
        "--feature-mode",
        choices=["minimal", "rich"],
        default="minimal",
    )
    p.add_argument(
        "--val-strategy",
        choices=["random", "stratified_additive", "none"],
        default="random",
    )
    p.add_argument(
        "--n-val-additives",
        type=int,
        default=None,
    )

    # Training-set-size ablation
    p.add_argument(
        "--train-size-ablation",
        action="store_true",
    )
    p.add_argument(
        "--train-size-fractions",
        type=float,
        nargs="+",
        default=[0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70],
    )
    p.add_argument(
        "--train-size-min-val",
        type=int,
        default=50,
    )

    args = p.parse_args()
    if args.quick:
        args.epochs = 10
        args.patience = 5
        args.seeds = 1
        args.batch_size = 8
        args.val_every = (
            1  # quick mode: validate every epoch so the smoke is informative
        )
        if args.train_size_ablation:
            args.train_size_fractions = [0.70]
    return args


def _confirm_device(args) -> str:
    """Catch the silent-fall-back-to-CPU case. Returns the resolved device."""
    device = args.device
    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "  WARNING: --device cuda specified but CUDA is unavailable; "
                "falling back to CPU."
            )
            return "cpu"
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    elif device == "cpu":
        print("  Running on CPU.")
    return device


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _confirm_device(args)
    print(f"=== bh_benchmark.py ===")
    print(f"  protocol = {args.protocol}")
    print(f"  device   = {device}")
    print(f"  out-dir  = {out_dir}")
    print(f"  feature-mode  = {args.feature_mode}")
    print(f"  val-strategy  = {args.val_strategy}")
    print(f"  val-every     = {args.val_every}")

    # Load split
    if args.protocol == "random":
        items = load_bh_full(Path(args.data_root), feature_mode=args.feature_mode)
        train, val, test = random_split(
            items,
            seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    else:
        sheet_map = {f"test{i}": f"Test{i}" for i in range(1, 5)}
        sheet_map.update({f"fullcv_{i:02d}": f"FullCV_{i:02d}" for i in range(1, 11)})
        sheet_name = sheet_map[args.protocol]
        train_full, test = load_bh_split(
            Path(args.data_root), sheet_name, feature_mode=args.feature_mode
        )

        if args.val_strategy == "random":
            train, val = val_split_from_train(
                train_full,
                seed=args.split_seed,
                val_fraction=args.val_fraction_from_train,
            )
        elif args.val_strategy == "stratified_additive":
            train, val = val_split_stratified_by_additive(
                train_full,
                seed=args.split_seed,
                n_val_additives=args.n_val_additives,
                val_fraction=args.val_fraction_from_train,
            )
        elif args.val_strategy == "none":
            train, val = train_full, []
        else:
            raise ValueError(f"Unknown --val-strategy: {args.val_strategy!r}")

    print(f"  train/val/test = {len(train)}/{len(val)}/{len(test)}")

    # Pre-warm the per-reaction packing cache so we don't pay the cost on the
    # first batch.
    t_pack = time.time()
    prewarm_pack_cache(train, val, test)
    print(f"  pre-packed reactions in {time.time() - t_pack:.2f}s")

    stats = compute_target_stats(train)
    stats_device = stats.to(device)
    print(
        f"  target stats (train): "
        f"mean={stats.mean.item():.2f} std={stats.std.item():.2f}"
    )

    hiddens, params_by_key = equalize_models(args)
    print(f"\n[equalize-params={args.equalize_params}] per-model budgets:")
    for k in args.models:
        print(
            f"  {MODEL_KEYS[k]:<32} hidden={hiddens[k]:>4}  "
            f"params={params_by_key[k]:>9,}"
        )

    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "models": {k: MODEL_KEYS[k] for k in args.models},
            "model_param_counts": params_by_key,
            "model_hidden_dims": hiddens,
            "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "target_stats": {"mean": stats.mean.item(), "std": stats.std.item()},
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cuda": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
        },
    )

    raw_scores, timings, flat_rows, history = train_eval_models(
        args,
        train,
        val,
        test,
        stats_device,
        hiddens,
        device,
    )

    per_model: Dict[str, Dict] = {}
    for mkey in args.models:
        s = raw_scores[mkey]
        if not s["mae"]:
            continue
        agg: Dict[str, Dict] = {}
        for metric in ("mae", "rmse", "r2"):
            mean, half = t_ci_95(s[metric])
            agg[metric] = {"mean": mean, "ci95": half, "scores": s[metric]}
        agg["params"] = params_by_key[mkey]
        agg["hidden"] = hiddens[mkey]
        agg["wall_seconds_mean"] = float(np.mean(timings[mkey]))
        per_model[mkey] = agg

    pairwise_raw = {}
    active = [k for k in args.models if len(raw_scores[k]["rmse"]) >= 2]
    for k1, k2 in combinations(active, 2):
        a = raw_scores[k1]["rmse"]
        b = raw_scores[k2]["rmse"]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        p = wilcoxon_signed_rank(a, b)
        d = cohens_d_paired(a, b)
        pairwise_raw[f"{MODEL_KEYS[k1]} vs {MODEL_KEYS[k2]}"] = (p, d)
    adj = holm_bonferroni({k: v[0] for k, v in pairwise_raw.items()})
    pairwise_rmse = {
        k: {"p_raw": p, "p_adj": adj[k][0], "reject": bool(adj[k][1]), "d": d}
        for k, (p, d) in pairwise_raw.items()
    }

    summary = {
        "metric_primary": "rmse",
        "per_model": per_model,
        "pairwise_rmse": pairwise_rmse,
    }

    write_json(
        out_dir / "raw.json",
        {
            "summary": summary,
            "raw_scores": raw_scores,
            "timings": timings,
            "history": history,
        },
    )
    write_csv(out_dir / "summary.csv", flat_rows)
    write_markdown(out_dir / "summary.md", summary, args, hiddens)
    write_latex(out_dir / "summary.tex", summary, args, hiddens)
    write_pairwise_latex(out_dir / "pairwise.tex", summary, args)

    print(f"\n========== summary ==========")
    r2s = {k: pm["r2"]["mean"] for k, pm in per_model.items()}
    valid = [v for v in r2s.values() if not math.isnan(v)]
    best_r2 = max(valid) if valid else float("nan")
    for mkey, pm in per_model.items():
        marker = " *" if abs(pm["r2"]["mean"] - best_r2) < 1e-9 else "  "
        print(
            f"{marker} {MODEL_KEYS[mkey]:<32} "
            f"hidden={hiddens[mkey]:>4}  "
            f"RMSE {pm['rmse']['mean']:.3f}±{pm['rmse']['ci95']:.3f}  "
            f"R² {pm['r2']['mean']:.4f}±{pm['r2']['ci95']:.4f}  "
            f"({pm['params']:,} params)"
        )
    print(f"\nAll outputs in: {out_dir}")


def write_learning_curve_csv(path: Path, rows: List[Dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    front = [k for k in ("train_fraction", "train_size") if k in keys]
    keys = front + [k for k in keys if k not in front]
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def write_learning_curve_md(path: Path, summary_by_fraction: Dict, args):
    out = ["# BH yield — training-set-size ablation", ""]
    out += [
        f"- Test set: fixed 30% of full dataset (split-seed {args.split_seed})",
        f"- Train fractions: {args.train_size_fractions}",
        f"- Seeds per fraction: {args.seeds}",
        f"- Epochs (max): {args.epochs}, patience: {args.patience}, val-every: {args.val_every}",
        f"- Feature mode: `{args.feature_mode}`",
        f"- Val strategy: `{args.val_strategy}`",
        f"- Min train size for val: {args.train_size_min_val}",
        f"- Equalization mode: `{args.equalize_params}`",
        "",
    ]
    for frac, block in summary_by_fraction.items():
        n_train = block["train_size"]
        out += [
            f"## Train fraction = {frac:.3f}  (n_train ≈ {n_train})",
            "",
            "| model | RMSE (%) | MAE (%) | R² |",
            "|---|---|---|---|",
        ]
        per_model = block["per_model"]
        valid_r2s = [
            pm["r2"]["mean"]
            for pm in per_model.values()
            if not math.isnan(pm["r2"]["mean"])
        ]
        best = max(valid_r2s) if valid_r2s else float("nan")
        for mkey, pm in per_model.items():
            rmse_s = (
                f"{pm['rmse']['mean']:.3f} ± {pm['rmse']['ci95']:.3f}"
                if not math.isnan(pm["rmse"]["mean"])
                else "n/a"
            )
            mae_s = (
                f"{pm['mae']['mean']:.3f} ± {pm['mae']['ci95']:.3f}"
                if not math.isnan(pm["mae"]["mean"])
                else "n/a"
            )
            r2_s = (
                f"{pm['r2']['mean']:.3f} ± {pm['r2']['ci95']:.3f}"
                if not math.isnan(pm["r2"]["mean"])
                else "n/a"
            )
            if not math.isnan(pm["r2"]["mean"]) and abs(pm["r2"]["mean"] - best) < 1e-9:
                r2_s = f"**{r2_s}**"
            out.append(f"| {MODEL_KEYS[mkey]} | {rmse_s} | {mae_s} | {r2_s} |")
        out.append("")
    path.write_text("\n".join(out) + "\n")


def plot_learning_curve(out_dir: Path, summary_by_fraction: Dict, args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fractions = sorted(summary_by_fraction.keys())
    if not fractions:
        return
    train_sizes = [summary_by_fraction[f]["train_size"] for f in fractions]

    color_cycle = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["C0", "C1", "C2", "C3", "C4"])
    )

    for metric, ylabel, fname in (
        ("r2", "R²", "learning_curve_r2.png"),
        ("rmse", "RMSE (% yield)", "learning_curve_rmse.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for ci, mkey in enumerate(args.models):
            color = color_cycle[ci % len(color_cycle)]
            ys = []
            errs = []
            for f in fractions:
                pm = summary_by_fraction[f]["per_model"].get(mkey)
                if pm is None:
                    ys.append(float("nan"))
                    errs.append(0.0)
                else:
                    ys.append(pm[metric]["mean"])
                    errs.append(
                        pm[metric]["ci95"]
                        if not math.isnan(pm[metric]["ci95"])
                        else 0.0
                    )
            ax.errorbar(
                train_sizes,
                ys,
                yerr=errs,
                label=MODEL_KEYS[mkey],
                color=color,
                marker="o",
                linewidth=1.6,
                capsize=3,
            )
        ax.set_xscale("log")
        ax.set_xlabel("training set size (reactions)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"BH yield — {ylabel} vs training-set size")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)


def run_train_size_ablation(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _confirm_device(args)
    print(f"=== bh_benchmark.py (train-size-ablation) ===")
    print(f"  device   = {device}")
    print(f"  out-dir  = {out_dir}")
    print(f"  fractions = {args.train_size_fractions}")
    print(f"  seeds per fraction = {args.seeds}")
    print(f"  val-every = {args.val_every}")

    items = load_bh_full(Path(args.data_root), feature_mode=args.feature_mode)
    n_full = len(items)
    print(f"  full dataset = {n_full} reactions")

    train_pool, _val_pool_unused, test = random_split(
        items,
        seed=args.split_seed,
        train_ratio=0.70,
        val_ratio=0.0,
    )
    n_pool = len(train_pool)
    n_test = len(test)
    print(f"  fixed test set = {n_test}, train pool = {n_pool}")

    # Pre-warm packing cache for the entire pool + test (val subsamples will
    # come from the pool, so no extra prewarm needed).
    t_pack = time.time()
    prewarm_pack_cache(train_pool, test)
    print(f"  pre-packed reactions in {time.time() - t_pack:.2f}s")

    stats = compute_target_stats(train_pool)
    stats_device = stats.to(device)
    print(
        f"  target stats (train pool): "
        f"mean={stats.mean.item():.2f} std={stats.std.item():.2f}"
    )

    hiddens, params_by_key = equalize_models(args)
    print(f"\n[equalize-params={args.equalize_params}] per-model budgets:")
    for k in args.models:
        print(
            f"  {MODEL_KEYS[k]:<32} hidden={hiddens[k]:>4}  "
            f"params={params_by_key[k]:>9,}"
        )

    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "models": {k: MODEL_KEYS[k] for k in args.models},
            "model_param_counts": params_by_key,
            "model_hidden_dims": hiddens,
            "n_full": n_full,
            "n_test": n_test,
            "n_pool": n_pool,
            "fractions": args.train_size_fractions,
            "target_stats": {"mean": stats.mean.item(), "std": stats.std.item()},
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cuda": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
        },
    )

    all_flat_rows: List[Dict] = []
    summary_by_fraction: Dict[float, Dict] = {}

    inner_seeds = args.seeds
    inner_seed_offset = args.seed_offset
    args.seeds = 1

    try:
        for frac in args.train_size_fractions:
            target_size = max(1, round(frac * n_full))
            target_size = min(target_size, n_pool)
            print(
                f"\n========== fraction={frac:.3f}  "
                f"target_size={target_size} (of pool {n_pool}) =========="
            )

            per_seed_raw = {k: {"mae": [], "rmse": [], "r2": []} for k in args.models}

            for outer_sidx in range(inner_seeds):
                outer_seed = inner_seed_offset + outer_sidx
                rng = random.Random(outer_seed * 1000 + int(frac * 1e6))
                pool_idx = list(range(n_pool))
                rng.shuffle(pool_idx)
                subset = [train_pool[i] for i in pool_idx[:target_size]]

                if target_size >= args.train_size_min_val:
                    val_n = max(
                        1, min(50, int(target_size * args.val_fraction_from_train))
                    )
                    val = subset[:val_n]
                    train = subset[val_n:]
                else:
                    val = []
                    train = subset

                args.seed_offset = outer_seed

                print(
                    f"\n  --- seed={outer_seed} (train={len(train)}, "
                    f"val={len(val)}, test={n_test}) ---"
                )
                raw, timings, rows, _ = train_eval_models(
                    args,
                    train,
                    val,
                    test,
                    stats_device,
                    hiddens,
                    device,
                    protocol_label=f"train_frac_{frac:.3f}",
                    extra_row_fields={
                        "train_fraction": frac,
                        "train_size": len(train),
                        "val_size": len(val),
                    },
                )
                for mkey in args.models:
                    for metric in ("mae", "rmse", "r2"):
                        per_seed_raw[mkey][metric].extend(raw[mkey][metric])
                all_flat_rows.extend(rows)

            per_model: Dict[str, Dict] = {}
            for mkey in args.models:
                s = per_seed_raw[mkey]
                if not s["mae"]:
                    continue
                agg: Dict[str, Dict] = {}
                for metric in ("mae", "rmse", "r2"):
                    mean, half = t_ci_95(s[metric])
                    agg[metric] = {
                        "mean": mean,
                        "ci95": half,
                        "scores": s[metric],
                    }
                agg["params"] = params_by_key[mkey]
                agg["hidden"] = hiddens[mkey]
                per_model[mkey] = agg

            train_size_actual = target_size
            if target_size >= args.train_size_min_val:
                train_size_actual = target_size - max(
                    1, min(50, int(target_size * args.val_fraction_from_train))
                )
            summary_by_fraction[frac] = {
                "train_size": train_size_actual,
                "target_size": target_size,
                "per_model": per_model,
            }
    finally:
        args.seeds = inner_seeds
        args.seed_offset = inner_seed_offset

    write_json(
        out_dir / "raw.json",
        {
            "summary_by_fraction": summary_by_fraction,
            "all_rows": all_flat_rows,
            "config": {
                "fractions": args.train_size_fractions,
                "n_full": n_full,
                "n_test": n_test,
                "n_pool": n_pool,
            },
        },
    )
    write_learning_curve_csv(out_dir / "summary.csv", all_flat_rows)
    write_learning_curve_md(out_dir / "summary.md", summary_by_fraction, args)
    plot_learning_curve(out_dir, summary_by_fraction, args)

    print(f"\n========== ablation summary ==========")
    print(f"{'fraction':>10} {'n_train':>8}  ", end="")
    for mkey in args.models:
        print(f"{MODEL_KEYS[mkey][:14]:>16}", end="")
    print()
    for frac in sorted(summary_by_fraction.keys()):
        block = summary_by_fraction[frac]
        print(f"{frac:>10.3f} {block['train_size']:>8}  ", end="")
        for mkey in args.models:
            pm = block["per_model"].get(mkey)
            if pm is None:
                print(f"{'n/a':>16}", end="")
            else:
                r2 = pm["r2"]["mean"]
                print(f"  R²={r2:>+.3f}     ", end="")
        print()
    print(f"\nAll outputs in: {out_dir}")


def main():
    args = parse_args()
    if args.train_size_ablation:
        run_train_size_ablation(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
