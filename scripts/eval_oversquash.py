"""
oversquash_benchmark.py
=======================

Benchmark whether GraphSetConv (GSC) handles graph over-squashing better
than standard GNNs on a synthetic dumbbell-graph task.

Architectural matrix
--------------------
Each model receives one of two views of every example:

  uncut: one connected graph; bottleneck of B edges between adjacent clusters
  cut:   K disconnected graphs (the bottleneck edges removed)

  GCN-only             - uncut, single graph -> GCN -> mean-pool -> MLP
                         Sees the bottleneck. Degrades as B shrinks.
  GCN+DeepSets-cut     - cut, K graphs -> GCN per graph -> DeepSets pool
  GCN+SetTransformer-cut - cut, K graphs -> GCN per graph -> SetTransformer
  GraphSetConv-V1-cut  - cut, K graphs -> GSC blocks (broadcast set-context)
  GraphSetConv-V2-cut  - cut, K graphs -> GSC blocks (per-node cross-attn)

Cut models receive the same input regardless of B (cut already removes
all bottleneck edges). Only GCN-only's input depends on B; we use it
as the "control" curve showing how over-squashing degrades a vanilla GNN.

Headline experiment (--sweep B)
-------------------------------
Sweep B in {1, 2, 4, 8, inf}. Plot test MAE per architecture vs. 1/B.
Expected:
- GCN-only (uncut) curve rises sharply as B -> 1
- All cut models stay flat (their input doesn't depend on B)
- Among cut models, GSC variants should match or beat the pipeline
  set-heads; that's the architectural-comparison axis

Secondary (--sweep N): vary cluster size N for fixed B=1. Larger N =
worse over-squashing for GCN-only; cut models nearly flat.

Chain ablation (--sweep K): K=2, 4, 8 with B=1 between adjacent clusters.
Target = matching pairs between cluster 0 and cluster K-1, forcing info
to traverse K-1 bottlenecks. GCN-only should degrade sharply with K.

Outputs
-------
For each sweep, writes raw.json + summary.csv + summary.md and a headline
PNG showing test MAE (or R^2) vs. the swept axis.

Usage
-----
    # Smoke test
    python oversquash_benchmark.py --quick

    # Headline: dumbbell K=2, sweep B, 5 seeds
    python oversquash_benchmark.py --sweep B --K 2 --N 20 \\
        --B-values 1 2 4 8 inf --seeds 5 \\
        --out-dir results/oversquash_dumbbell_B

    # Cluster-size scaling at B=1
    python oversquash_benchmark.py --sweep N --K 2 --B 1 --N-values 10 20 40 --seeds 5 --out-dir results/oversquash_dumbbell_N

    # Chain ablation
    python oversquash_benchmark.py --sweep K --B 1 --N 10 \\
        --K-values 2 4 8 --seeds 5 \\
        --out-dir results/oversquash_chain_K
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oversquash_data import (
    NUM_LABELS,
    DumbbellSample,
    TargetStats,
    compute_target_stats,
    generate_dumbbell_dataset,
    random_split,
)

# Reuse blocks. Match the user's package layout.
try:
    from graph_set_transformer.models.gst import GraphSetConv
except ImportError:
    from graph_set_conv import GraphSetConv


# =============================================================================
# Pure-torch helpers (torch_scatter-free)
# =============================================================================


def global_add_pool_safe(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = x.new_zeros(n, x.size(-1))
    out.index_add_(0, batch, x)
    return out


def global_mean_pool_safe(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = x.new_zeros(n, x.size(-1))
    out.index_add_(0, batch, x)
    counts = x.new_zeros(n)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(-1)


# =============================================================================
# Set heads
# =============================================================================


class DeepSetsHead(nn.Module):
    def __init__(self, dim, dropout=0.1):
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

    def forward(self, z_graph, set_batch):
        h = self.phi(z_graph)
        z_set = global_add_pool_safe(h, set_batch)
        return self.rho(z_set)


class SetTransformerHead(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, ffn_mult=2):
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

    def forward(self, z_graph, set_batch):
        from torch_geometric.utils import to_dense_batch

        z_dense, mask = to_dense_batch(z_graph, set_batch)
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
# Models
# =============================================================================


# Each entry: (display_name, view) where view is "uncut" or "cut".
MODEL_KEYS = {
    "gcn-only": ("GCN-only (uncut)", "uncut"),
    "gcn-ds": ("GCN+DeepSets (cut)", "cut"),
    "gcn-st": ("GCN+SetTransformer (cut)", "cut"),
    "gsc-v1": ("GraphSetConv-V1 (cut)", "cut"),
    "gsc-v2": ("GraphSetConv-V2 (cut)", "cut"),
}


class GCNStack(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout=0.1):
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


class OversquashModel(nn.Module):
    """Five model variants. The 'view' is determined by model_key.

    For uncut models (gcn-only): batch_obj.x is the atoms across all
    samples in the minibatch; batch_obj.edge_index is the uncut graph
    edges; batch_obj.batch maps atom -> sample id (one graph per sample).

    For cut models: batch_obj.edge_index is the cut graph edges (no
    cross-cluster); batch_obj.batch maps atom -> graph id (K graphs per
    sample); batch_obj.set_batch maps graph id -> sample id.
    """

    def __init__(self, model_key, num_layers, hidden, num_heads, dropout):
        super().__init__()
        self.model_key = model_key
        _, view = MODEL_KEYS[model_key]
        self.view = view
        self.input_proj = nn.Linear(NUM_LABELS, hidden)

        if model_key == "gcn-only":
            self.encoder = GCNStack(hidden, hidden, num_layers, dropout=dropout)
            self.set_head = None
        elif model_key in ("gcn-ds", "gcn-st"):
            self.encoder = GCNStack(hidden, hidden, num_layers, dropout=dropout)
            if model_key == "gcn-ds":
                self.set_head = DeepSetsHead(hidden, dropout=dropout)
            else:
                self.set_head = SetTransformerHead(hidden, num_heads, dropout=dropout)
        else:  # gsc-v1, gsc-v2
            self.encoder = None
            blocks = []
            for _ in range(num_layers):
                mode = "broadcast" if model_key == "gsc-v1" else "cross_attn"
                blocks.append(
                    GraphSetConv(
                        filters=hidden,
                        in_channels=hidden,
                        num_heads=num_heads,
                        mhsa_dropout=dropout,
                        ffn_dropout=dropout,
                        node_set_mode=mode,
                    )
                )
            self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch_obj):
        x = self.input_proj(batch_obj.x)

        if self.model_key == "gcn-only":
            # batch_obj.batch is atom -> sample (one graph per sample)
            x = self.encoder(x, batch_obj.edge_index)
            z = global_mean_pool_safe(x, batch_obj.batch)
            return self.head(z).squeeze(-1)

        # cut models: batch_obj.batch is atom -> graph_id, set_batch is graph_id -> sample
        if self.encoder is not None:
            # pipeline: GCN encode then per-graph pool then set head
            x = self.encoder(x, batch_obj.edge_index)
            z_graph = global_mean_pool_safe(x, batch_obj.batch)
            z_set = self.set_head(z_graph, batch_obj.set_batch)
        else:
            # GSC: blocks operate on (atoms, edges, atom->graph, graph->set)
            for blk in self.blocks:
                x = blk(x, batch_obj.edge_index, batch_obj.batch, batch_obj.set_batch)
            z_graph = global_mean_pool_safe(x, batch_obj.batch)
            z_set = global_mean_pool_safe(z_graph, batch_obj.set_batch)
        return self.head(z_set).squeeze(-1)


def build_model(key, num_layers, hidden, num_heads, dropout):
    return OversquashModel(
        model_key=key,
        num_layers=num_layers,
        hidden=hidden,
        num_heads=num_heads,
        dropout=dropout,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Param equalization
# =============================================================================


# Group models for equalization scaling.
PIPELINE_KEYS = {"gcn-ds", "gcn-st"}
GSC_KEYS = {"gsc-v1", "gsc-v2"}
GCN_ONLY_KEY = "gcn-only"


def find_hidden_for_target_params(builder, target, num_heads, lo=8, hi=1024):
    """Scan candidate hidden dims to find the one whose param count is
    closest to `target`. Constrained to hidden % num_heads == 0 so MHA
    inside set heads / GSC stays valid. Stops early once we overshoot
    the target by 4× to avoid wasting time on huge models."""
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
    """Resolve per-model hidden dim and param count under the chosen
    equalization mode. Returns (hidden_per_model, params_per_model).

    none:    every model uses --hidden.
    smaller: pipelines stay at --hidden; GSC variants scale DOWN to match
             the smaller pipeline (GCN+DeepSets) param count. GCN-only
             stays at --hidden (it's already the smallest model).
    larger:  GSC variants stay at --hidden; pipelines scale UP to match
             the largest GSC variant. GCN-only also scales UP to match
             the largest GSC, so the over-squashing comparison isn't
             confounded by GCN-only being under-resourced.

    Note: 'smaller' tests "GSC at pipeline budget", which is the budget-
    fair version of the architectural claim. 'larger' tests "do simpler
    models catch up given enough capacity", which is the budget-generous
    version. For the over-squashing experiment, 'larger' is arguably the
    cleaner test — it removes the worry that GCN-only loses simply
    because it's smaller, not because of the bottleneck.
    """
    base_hidden = args.hidden

    def build_for(key, hidden):
        return build_model(key, args.layers, hidden, args.num_heads, args.dropout)

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
                "--equalize-params smaller requires at least one of "
                "gcn-ds or gcn-st in --models."
            )
        target_p = min(base_params[k] for k in active_pipelines)
        hiddens = dict(base_hiddens)
        params = dict(base_params)
        for k in args.models:
            if k in PIPELINE_KEYS or k == GCN_ONLY_KEY:
                continue  # pipelines and GCN-only stay at base_hidden
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
                "--equalize-params larger requires at least one of "
                "gsc-v1 or gsc-v2 in --models."
            )
        target_p = max(base_params[k] for k in active_gsc)
        hiddens = dict(base_hiddens)
        params = dict(base_params)
        for k in args.models:
            if k in GSC_KEYS:
                continue  # GSC stays at base_hidden
            # Both pipelines and GCN-only scale up to match the largest GSC.
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
# Batch packing
# =============================================================================


class PackedBatch:
    """Holds a packed minibatch. The two views require different layouts:

    For uncut models:
      x:           [N_total_atoms, NUM_LABELS]
      edge_index:  [2, E_uncut_total]
      batch:       [N_total_atoms]   atom -> sample_id
      set_batch:   None  (not used)
      y:           [S]

    For cut models:
      x:           [N_total_atoms, NUM_LABELS]    (same atoms as uncut)
      edge_index:  [2, E_cut_total]               (no cross-cluster edges)
      batch:       [N_total_atoms]   atom -> graph_id (K graphs per sample)
      set_batch:   [G_total]         graph_id -> sample_id
      y:           [S]
    """

    def __init__(self, x, edge_index, batch, set_batch, y):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.set_batch = set_batch
        self.y = y

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        if self.set_batch is not None:
            self.set_batch = self.set_batch.to(device)
        self.y = self.y.to(device)
        return self


def collate(samples: List[DumbbellSample], view: str) -> PackedBatch:
    """Pack a list of DumbbellSamples into a single PackedBatch.

    view: 'uncut' (for gcn-only) or 'cut' (for everything else).
    """
    xs, edges, batch_ids, set_ids, ys = [], [], [], [], []
    n_off = 0  # atom-index offset into the packed tensor
    g_off = 0  # graph-id offset (used in cut view)

    for sid, s in enumerate(samples):
        N_sample = s.x.shape[0]

        xs.append(s.x)

        if view == "uncut":
            # One graph per sample; batch maps atom -> sample_id
            ei = s.edge_index_uncut
            if ei.numel() > 0:
                edges.append(ei + n_off)
            batch_ids.append(torch.full((N_sample,), sid, dtype=torch.long))
        else:
            # K graphs per sample; batch maps atom -> graph_id, set_batch maps graph_id -> sample_id
            ei = s.edge_index_cut
            if ei.numel() > 0:
                edges.append(ei + n_off)
            # Each cluster k in this sample becomes graph_id (g_off + k)
            atom_to_local_graph = s.batch_cut_local
            batch_ids.append(atom_to_local_graph + g_off)
            for k in range(s.K):
                set_ids.append(torch.tensor([sid], dtype=torch.long))
            g_off += s.K

        ys.append(s.y)
        n_off += N_sample

    x = torch.cat(xs, dim=0)
    edge_index = (
        torch.cat(edges, dim=1) if edges else torch.zeros((2, 0), dtype=torch.long)
    )
    batch = torch.cat(batch_ids, dim=0)
    set_batch = torch.cat(set_ids, dim=0) if view == "cut" else None
    y = torch.tensor(ys, dtype=torch.float32)
    return PackedBatch(x, edge_index, batch, set_batch, y)


# =============================================================================
# Stats helpers (Wilcoxon + Holm)
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
    """Compute MAE, RMSE, R^2 on the un-normalized scale."""
    if not items:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    model.eval()
    view = MODEL_KEYS[model.model_key][1]
    preds, truths = [], []
    with torch.no_grad():
        for samples in iterate_minibatches(
            items, args.batch_size, random.Random(0), shuffle=False
        ):
            pb = collate(samples, view).to(device)
            pred_norm = model(pb)
            pred = stats.denormalize(pred_norm)
            preds.append(pred.cpu())
            truths.append(pb.y.cpu())
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    view = MODEL_KEYS[model.model_key][1]

    best_val = float("inf")
    best_state = None
    epochs_since_improvement = 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        for samples in iterate_minibatches(train, args.batch_size, rng):
            pb = collate(samples, view).to(device)
            y_norm = stats.normalize(pb.y)
            pred = model(pb)
            loss = F.l1_loss(pred, y_norm)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_finite = all(
                p.grad is None or torch.isfinite(p.grad).all().item()
                for p in model.parameters()
            )
            if not grad_finite:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss.item() * pb.y.size(0)
            running_n += pb.y.size(0)
        if running_n == 0:
            print(f"    [seed={seed} epoch={epoch}] all batches nan; aborting")
            break

        v = eval_set(model, val, stats, args, device)
        history.append({"epoch": epoch, "val_mae": v["mae"], "val_r2": v["r2"]})

        if math.isfinite(v["mae"]) and v["mae"] < best_val:
            best_val = v["mae"]
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = eval_set(model, test, stats, args, device)
    return test_metrics, time.time() - t_start, history


# =============================================================================
# Output writers + plot
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


def write_markdown(path, sweep_results, args):
    """sweep_results: list of dicts, one per sweep point; each dict has
    'config' and 'per_model'."""
    out = ["# Over-squashing benchmark", ""]
    out += [
        f"- Sweep axis: `{args.sweep}`",
        f"- Models: {len(args.models)}",
        f"- Seeds: {args.seeds}",
        f"- Hidden: {args.hidden}, Layers: {args.layers}",
        f"- Train/val/test = {args.train_size}/{args.val_size}/{args.test_size}",
        "",
    ]
    for entry in sweep_results:
        cfg = entry["config"]
        out.append(f"## Config: {cfg}")
        out.append("")
        out.append("| model | params | RMSE | MAE | R² |")
        out.append("|---|---:|---|---|---|")
        per_model = entry["per_model"]
        # Best by lowest mean RMSE
        rmse_means = {
            k: v["rmse"]["mean"]
            for k, v in per_model.items()
            if not math.isnan(v["rmse"]["mean"])
        }
        best = min(rmse_means, key=rmse_means.get) if rmse_means else None
        for mkey, pm in per_model.items():
            r = pm["rmse"]
            m = pm["mae"]
            rr = pm["r2"]
            r_s = (
                f"{r['mean']:.4f} ± {r['ci95']:.4f}"
                if not math.isnan(r["mean"])
                else "n/a"
            )
            m_s = (
                f"{m['mean']:.4f} ± {m['ci95']:.4f}"
                if not math.isnan(m["mean"])
                else "n/a"
            )
            rr_s = (
                f"{rr['mean']:.4f} ± {rr['ci95']:.4f}"
                if not math.isnan(rr["mean"])
                else "n/a"
            )
            if mkey == best:
                r_s = f"**{r_s}**"
            out.append(
                f"| {MODEL_KEYS[mkey][0]} | {pm['params']:,} | {r_s} | {m_s} | {rr_s} |"
            )
        out.append("")
    path.write_text("\n".join(out) + "\n")


def plot_sweep(sweep_results, args, out_dir):
    """Plot test MAE (and R²) vs. the swept axis. One line per model."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [oversquash] matplotlib not available; skipping plot")
        return

    if not sweep_results:
        return

    # Extract sweep x-values from configs.
    sweep_axis = args.sweep
    xs = []
    for entry in sweep_results:
        cfg = entry["config"]
        if sweep_axis == "B":
            v = cfg["B"]
            xs.append(v if v != math.inf else float(args.B_inf_marker))
        elif sweep_axis == "N":
            xs.append(cfg["N"])
        elif sweep_axis == "K":
            xs.append(cfg["K"])

    # For each metric, draw lines for each model.
    for metric in ("mae", "r2"):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        color_cycle = (
            plt.rcParams["axes.prop_cycle"]
            .by_key()
            .get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])
        )
        for color_i, mkey in enumerate(args.models):
            ys, errs = [], []
            for entry in sweep_results:
                pm = entry["per_model"].get(mkey)
                if pm is None:
                    ys.append(float("nan"))
                    errs.append(0.0)
                    continue
                ys.append(pm[metric]["mean"])
                errs.append(
                    pm[metric]["ci95"] if not math.isnan(pm[metric]["ci95"]) else 0.0
                )
            ys = np.array(ys)
            errs = np.array(errs)
            color = color_cycle[color_i % len(color_cycle)]
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                label=MODEL_KEYS[mkey][0],
                marker="o",
                capsize=3,
                color=color,
                linewidth=1.6,
            )
        ax.set_xlabel(f"sweep axis: {sweep_axis}")
        ax.set_ylabel(metric)
        title_suffix = ""
        if sweep_axis == "B":
            title_suffix = "  (lower B = tighter bottleneck)"
        ax.set_title(f"Test {metric} vs. {sweep_axis}{title_suffix}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        # If sweep is B and uses an "inf" marker, annotate the rightmost point
        if sweep_axis == "B" and any(
            entry["config"].get("B") == math.inf for entry in sweep_results
        ):
            ax.annotate(
                "(B=inf, fully bipartite)",
                xy=(xs[-1], ax.get_ylim()[1] * 0.95),
                fontsize=7,
                ha="right",
                color="gray",
            )
        fig.tight_layout()
        fname = f"sweep_{sweep_axis}_{metric}.png"
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)
    print(f"  [oversquash] wrote sweep plots to {out_dir}")


# =============================================================================
# Main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/oversquash_run")
    p.add_argument("--quick", action="store_true")

    p.add_argument(
        "--sweep",
        choices=["B", "N", "K"],
        default="B",
        help="Which axis to sweep. B = bottleneck width. "
        "N = cluster size. K = number of clusters.",
    )
    p.add_argument(
        "--B-values",
        nargs="+",
        default=["1", "2", "4", "8", "inf"],
        help="Bottleneck widths to sweep (when --sweep=B). "
        "Use 'inf' for fully bipartite (no bottleneck).",
    )
    p.add_argument(
        "--N-values",
        nargs="+",
        type=int,
        default=[10, 20, 40],
        help="Cluster sizes to sweep (when --sweep=N).",
    )
    p.add_argument(
        "--K-values",
        nargs="+",
        type=int,
        default=[2, 4, 8],
        help="Cluster counts to sweep (when --sweep=K).",
    )

    # Fixed values (for the axes not being swept)
    p.add_argument(
        "--K",
        type=int,
        default=2,
        help="Number of clusters when not swept. K=2 = dumbbell.",
    )
    p.add_argument("--N", type=int, default=20, help="Cluster size when not swept.")
    p.add_argument("--B", type=int, default=1, help="Bottleneck width when not swept.")
    p.add_argument(
        "--intra-p",
        type=float,
        default=0.5,
        help="Erdős–Rényi probability within each cluster.",
    )

    # Dataset sizes
    p.add_argument("--train-size", type=int, default=4000)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--data-seed", type=int, default=0)

    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_KEYS.keys()),
        choices=list(MODEL_KEYS.keys()),
    )

    # Architecture
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--equalize-params",
        choices=["none", "smaller", "larger"],
        default="none",
        help="Param equalization across models. 'none' = every model uses "
        "--hidden. 'smaller' = pipelines and GCN-only stay at --hidden; "
        "GSC scales DOWN to match the smaller pipeline (GCN+DeepSets). "
        "'larger' = GSC stays at --hidden; pipelines AND GCN-only scale "
        "UP to match the largest GSC variant. For over-squashing "
        "experiments, 'larger' is the cleaner test (removes the worry "
        "that GCN-only loses simply because it's under-parameterized).",
    )

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)

    # Plotting
    p.add_argument(
        "--B-inf-marker",
        type=float,
        default=16.0,
        help="What x-position to plot 'B=inf' at on the B-sweep "
        "plot. Visual choice only.",
    )

    args = p.parse_args()
    if args.quick:
        args.epochs = 5
        args.patience = 5
        args.seeds = 1
        args.train_size = 200
        args.val_size = 50
        args.test_size = 100
        args.batch_size = 16
        args.B_values = ["1", "8"]
        args.N_values = [10, 20]
        args.K_values = [2, 4]
    return args


def parse_B(s: str) -> float:
    if s.lower() == "inf":
        return math.inf
    return int(s)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"=== oversquash_benchmark.py ===")
    print(f"  device  = {device}")
    print(f"  out-dir = {out_dir}")
    print(f"  sweep   = {args.sweep}")

    # Build sweep configurations.
    if args.sweep == "B":
        configs = [{"K": args.K, "N": args.N, "B": parse_B(s)} for s in args.B_values]
    elif args.sweep == "N":
        configs = [{"K": args.K, "N": n, "B": args.B} for n in args.N_values]
    elif args.sweep == "K":
        configs = [{"K": k, "N": args.N, "B": args.B} for k in args.K_values]
    else:
        raise ValueError(f"unknown sweep: {args.sweep}")

    # Resolve per-model hidden dim under equalization. Same architecture
    # for all sweep configs (K/N/B affect the data, not the model).
    hiddens, params_by_key = equalize_models(args)
    print(f"\n[equalize-params={args.equalize_params}] per-model budgets:")
    for k in args.models:
        print(
            f"  {MODEL_KEYS[k][0]:<32} hidden={hiddens[k]:>4}  "
            f"params={params_by_key[k]:>9,}"
        )

    # Persist run config
    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "models": {k: MODEL_KEYS[k][0] for k in args.models},
            "model_hidden_dims": hiddens,
            "model_param_counts": params_by_key,
            "configs": configs,
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

    sweep_results = []
    flat_rows = []

    for cfg_i, cfg in enumerate(configs):
        K, N, B = cfg["K"], cfg["N"], cfg["B"]
        # For B=inf, use a very large bottleneck width = N*N (fully bipartite).
        B_use = N * N if (isinstance(B, float) and math.isinf(B)) else int(B)
        print(f"\n[{cfg_i + 1}/{len(configs)}] K={K} N={N} B={B} (B_use={B_use})")

        # Generate dataset for this config
        all_items = generate_dumbbell_dataset(
            n_samples=args.train_size + args.val_size + args.test_size,
            K=K,
            N=N,
            B=B_use,
            intra_p=args.intra_p,
            num_labels=NUM_LABELS,
            seed=args.data_seed,
        )
        train, val, test = random_split(
            all_items,
            seed=args.data_seed,
            train_ratio=args.train_size / len(all_items),
            val_ratio=args.val_size / len(all_items),
        )
        # Force exact sizes (rounding can produce off-by-one)
        train = train[: args.train_size]
        val = val[: args.val_size]
        test = test[: args.test_size]
        print(f"  train/val/test = {len(train)}/{len(val)}/{len(test)}")

        stats = compute_target_stats(train)
        stats_device = stats.to(device)
        ymean = stats.mean.item()
        ystd = stats.std.item()
        print(f"  target stats: mean={ymean:.4f}, std={ystd:.4f}")

        per_model_results = {}
        for mkey in args.models:
            display, view = MODEL_KEYS[mkey]
            print(f"\n  --- {display} (view={view}) ---")
            seed_metrics = {"mae": [], "rmse": [], "r2": []}
            seed_times = []
            n_params_recorded = None
            for sidx in range(args.seeds):
                seed = args.seed_offset + sidx
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                model = build_model(
                    mkey, args.layers, hiddens[mkey], args.num_heads, args.dropout
                ).to(device)
                n_params = count_params(model)
                n_params_recorded = n_params
                metrics, elapsed, _ = train_one(
                    model,
                    train,
                    val,
                    test,
                    stats_device,
                    args,
                    seed,
                    device,
                )
                print(
                    f"    seed={seed:>2}  RMSE={metrics['rmse']:.4f}  "
                    f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}  "
                    f"({elapsed:.1f}s, {n_params:,} params)"
                )
                for k in ("mae", "rmse", "r2"):
                    seed_metrics[k].append(metrics[k])
                seed_times.append(elapsed)
                flat_rows.append(
                    {
                        "K": K,
                        "N": N,
                        "B": str(B),
                        "model_key": mkey,
                        "model_name": display,
                        "view": view,
                        "seed": seed,
                        "hidden": hiddens[mkey],
                        "params": n_params,
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "r2": metrics["r2"],
                        "wall_seconds": round(elapsed, 2),
                    }
                )
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            agg = {}
            for metric in ("mae", "rmse", "r2"):
                mean, half = t_ci_95(seed_metrics[metric])
                agg[metric] = {
                    "mean": mean,
                    "ci95": half,
                    "scores": seed_metrics[metric],
                }
            agg["params"] = n_params_recorded
            agg["wall_seconds_mean"] = float(np.mean(seed_times))
            per_model_results[mkey] = agg

        sweep_results.append({"config": cfg, "per_model": per_model_results})

        # Console summary for this config
        print(f"\n  ===== summary for K={K} N={N} B={B} =====")
        rmse_means = {
            k: v["rmse"]["mean"]
            for k, v in per_model_results.items()
            if not math.isnan(v["rmse"]["mean"])
        }
        best = min(rmse_means, key=rmse_means.get) if rmse_means else None
        for mkey, pm in per_model_results.items():
            marker = " *" if mkey == best else "  "
            print(
                f"  {marker} {MODEL_KEYS[mkey][0]:<32} "
                f"RMSE {pm['rmse']['mean']:.4f}±{pm['rmse']['ci95']:.4f}  "
                f"R² {pm['r2']['mean']:.4f}±{pm['r2']['ci95']:.4f}  "
                f"({pm['params']:,} params)"
            )

    # Persist all results
    write_json(
        out_dir / "raw.json",
        {
            "sweep_axis": args.sweep,
            "configs": [r["config"] for r in sweep_results],
            "per_config_per_model": [
                {"config": r["config"], "per_model": r["per_model"]}
                for r in sweep_results
            ],
        },
    )
    write_csv(out_dir / "summary.csv", flat_rows)
    write_markdown(out_dir / "summary.md", sweep_results, args)
    plot_sweep(sweep_results, args, out_dir)

    print(f"\nAll outputs in: {out_dir}")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
