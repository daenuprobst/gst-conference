"""
uspto15k_benchmark.py
=====================

Real-world counterpart to the RING-K diagnostic: per-atom binary classification
of *reaction centers* on USPTO-15K (Jin et al., NeurIPS 2017). Each example is
a chemical reaction; each reactant molecule is a graph; the input is the *set*
of reactant graphs; the per-atom label is 1 iff that atom participates in the
reaction center (its bonds change going to the product).

This is the chemical instantiation of RING-K: positives are sparse (~5% of
atoms; Jin et al. 2017), and whether a given atom is positive depends not only
on its local environment within its own molecule but on which other molecules
are present in the reaction. A different reagent or co-reactant in the same
flask produces a different reaction center on the same substrate.

The model registry is identical to synth_benchmark.py: GraphSetConv-CrossAttn,
GraphSetConv-Broadcast, GCN+DeepSets, GCN+SetTransformer. We are NOT chasing
SOTA against task-specific baselines (WLN, GRAPHRETRO, MARCC, RCsearcher);
this is an architecture ablation under matched parameter budgets.

Data
----
Expected at --data-dir:
    train.txt      reaction lines (rxn_smi, edits)
    valid.txt
    test.txt

Each line has two whitespace-separated fields:
    rxn_smi        atom-mapped reaction SMILES, "reactants>>product"
    edits          four ';'-separated subfields:
                       atoms-lost-H ; atoms-gained-H ; deleted-bonds ; added-bonds
                   atoms are atom-map numbers; bonds are "a-b-type".

The Jin et al. release ships these as USPTO-15K/data.zip in
github.com/wengong-jin/nips17-rexgen ; the user is expected to unpack that
into --data-dir before running.

Methodology references
----------------------
- Reaction-center identification: Jin, Coley, Barzilay & Jaakkola, "Predicting
  Organic Reaction Outcomes with Weisfeiler-Lehman Network" (NeurIPS 2017).
  Per-reactant atom labels derived from the four edit lists released with the
  USPTO-15K split.
- Set-conditional framing: each reactant molecule is a graph; the reaction is
  the set of those graphs; the per-atom label is set-conditional because the
  reagents present co-determine which atoms react.
- Statistics, parameter equalisation, model registry: identical to
  synth_benchmark.py (Wilcoxon signed-rank with Holm-Bonferroni; Cohen's d_z;
  Student-t 95% CIs; --equalize-params {none,smaller,larger}).

Usage
-----
Quick smoke test:
    python uspto15k_benchmark.py --data-dir USPTO-15K --quick

Full publication run:
    python uspto15k_benchmark.py --data-dir USPTO-15K \\
        --equalize-params smaller --seeds 5 --out-dir results/uspto15k

Requires graph_set_conv.py (or graph_set_transformer.models.gst_v2.GraphSetConv)
on PYTHONPATH, plus rdkit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import (
    GCNConv,
    GraphNorm,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit warnings on minor SMILES quirks

# Local import: same GraphSetConv block used by the synthetic benchmark
from graph_set_transformer.models.gst import GraphSetConv  # noqa: E402


# ============================================================================
# Data loading: parse USPTO-15K reaction lines into per-atom labels
# ============================================================================


@dataclass
class ReactionExample:
    """One reaction parsed from a USPTO-15K line.

    Each entry of `mols`, `atom_maps`, `labels`, `node_features`, `edges`
    refers to one reactant molecule. The reaction is the *set* of these.
    """

    rxn_id: int
    mols: List[Chem.Mol]  # one RDKit Mol per reactant
    atom_maps: List[List[int]]  # per-atom map numbers, parallel to mol atoms
    labels: List[np.ndarray]  # per-atom binary {0,1}
    node_features: List[np.ndarray]  # per-atom feature matrices [n_atoms, F]
    edges: List[np.ndarray]  # [2, 2*E] undirected edge_index per molecule


# Atom features are deliberately minimal: enough chemistry to distinguish
# elements and basic bonding context, but not so much that the model can
# trivially identify common reaction-centre atoms by element alone (e.g.,
# 'every Cl is a reaction centre' shortcut). Anything stronger would be a
# task-specific feature engineering decision and is out of scope for an
# architecture ablation.
ATOM_FEATURE_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]


def _atom_features(atom: Chem.Atom) -> np.ndarray:
    """Per-atom feature vector. Length = len(ATOM_FEATURE_ELEMENTS) + 7."""
    one_hot_el = [
        1.0 if atom.GetSymbol() == el else 0.0 for el in ATOM_FEATURE_ELEMENTS
    ]
    other_el = 0.0 if any(one_hot_el) else 1.0
    feats = one_hot_el + [
        other_el,
        float(atom.GetDegree()) / 6.0,
        float(atom.GetFormalCharge()),
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetTotalNumHs()) / 4.0,
        1.0 if atom.IsInRing() else 0.0,
        float(int(atom.GetHybridization())) / 7.0,
    ]
    return np.array(feats, dtype=np.float32)


def atom_feature_dim() -> int:
    return len(ATOM_FEATURE_ELEMENTS) + 7


def _mol_to_arrays(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    n = mol.GetNumAtoms()
    feats = np.stack([_atom_features(a) for a in mol.GetAtoms()], axis=0)
    amaps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if mol.GetNumBonds() == 0:
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        rows, cols = [], []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            rows.extend([i, j])
            cols.extend([j, i])
        ei = np.array([rows, cols], dtype=np.int64)
    return feats, ei, amaps


def _parse_atom_list(s: str) -> set:
    return set(int(x) for x in s.split(",") if x.strip())


def _parse_bond_list(s: str) -> List[Tuple[int, int, float]]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split("-")
        if len(parts) != 3:
            continue  # malformed; skip silently rather than crash on one bad line
        a, b, t = parts
        try:
            out.append((int(a), int(b), float(t)))
        except ValueError:
            continue
    return out


def _parse_edits(edits: str) -> set:
    """Return set of atom-map numbers participating in any edit."""
    parts = edits.split(";")
    if len(parts) != 4:
        return set()  # malformed; downstream caller will drop the example
    h_lost, h_gain, b_del, b_add = parts
    rc = set()
    rc |= _parse_atom_list(h_lost)
    rc |= _parse_atom_list(h_gain)
    for a, b, _t in _parse_bond_list(b_del) + _parse_bond_list(b_add):
        rc.add(a)
        rc.add(b)
    return rc


def _parse_line(rxn_id: int, line: str) -> Optional[ReactionExample]:
    """Parse one USPTO-15K data line; return None if anything fails."""
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    rxn_smi, edits_field = parts[0], parts[1]
    try:
        reactants_smi = rxn_smi.split(">>")[0]
    except Exception:
        return None
    rc_atoms = _parse_edits(edits_field)
    if not rc_atoms:
        return None  # we cannot learn from a reaction with no parseable edits

    mols: List[Chem.Mol] = []
    atom_maps: List[List[int]] = []
    feats_list: List[np.ndarray] = []
    edges_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for sub in reactants_smi.split("."):
        sub = sub.strip()
        if not sub:
            continue
        m = Chem.MolFromSmiles(sub)
        if m is None or m.GetNumAtoms() == 0:
            return None  # any unparseable reactant invalidates the example
        feats, ei, amaps = _mol_to_arrays(m)
        labels = np.array(
            [1.0 if amap in rc_atoms else 0.0 for amap in amaps], dtype=np.float32
        )
        mols.append(m)
        atom_maps.append(amaps)
        feats_list.append(feats)
        edges_list.append(ei)
        labels_list.append(labels)

    if not mols:
        return None
    if all(lbl.sum() == 0 for lbl in labels_list):
        # The atom-map numbers in the edits don't match any reactant atoms.
        # Treat as malformed rather than as a true all-negative example.
        return None
    return ReactionExample(
        rxn_id=rxn_id,
        mols=mols,
        atom_maps=atom_maps,
        labels=labels_list,
        node_features=feats_list,
        edges=edges_list,
    )


def load_split(path: Path, max_examples: Optional[int] = None) -> List[ReactionExample]:
    examples: List[ReactionExample] = []
    n_seen = n_kept = 0
    with open(path, "r") as f:
        for i, line in enumerate(f):
            n_seen += 1
            ex = _parse_line(i, line)
            if ex is not None:
                examples.append(ex)
                n_kept += 1
                if max_examples is not None and n_kept >= max_examples:
                    break
    print(
        f"  loaded {n_kept}/{n_seen} examples from {path.name} "
        f"({100 * n_kept / max(n_seen, 1):.1f}% kept)"
    )
    return examples


# ============================================================================
# Batching
# ============================================================================


@dataclass
class SetBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    set_batch: torch.Tensor
    y: torch.Tensor
    n_sets: int
    n_graphs: int
    n_nodes: int


def collate(samples: List[ReactionExample], device: str) -> SetBatch:
    """Pack a list of reactions into one PyG-style batched object.

    `batch` indexes per-graph (one id per molecule across the whole batch);
    `set_batch` indexes per-set (one id per graph indicating its reaction).
    Each reaction is one set; nodes within a reaction are the atoms of all
    its reactants.
    """
    xs, edge_indices, batch_ids, set_ids, ys = [], [], [], [], []
    n_offset, g_offset, s_offset = 0, 0, 0
    for ex in samples:
        for feats, ei, lbl in zip(ex.node_features, ex.edges, ex.labels):
            n = feats.shape[0]
            xs.append(torch.from_numpy(feats))
            ys.append(torch.from_numpy(lbl))
            ei_t = (
                torch.from_numpy(ei) + n_offset
                if ei.shape[1] > 0
                else (torch.zeros((2, 0), dtype=torch.long))
            )
            edge_indices.append(ei_t)
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


# ============================================================================
# Models (identical registry to synth_benchmark.py)
# ============================================================================


def make_activation():
    return nn.SiLU()


class InputProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)

    def forward(self, x):
        return self.proj(x)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout=0.1):
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
            x = x + h
        return x


class DeepSetsHead(nn.Module):
    def __init__(self, hidden, dropout=0.1):
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
        h = self.phi(z_graph)
        z_set = global_add_pool(h, set_batch)
        return self.rho(z_set)


class SetTransformerHead(nn.Module):
    def __init__(self, hidden, num_heads, dropout=0.1, ffn_mult=2):
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
        self.seed = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.ln3 = nn.LayerNorm(hidden)
        self.pma = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, z_graph, set_batch):
        z_dense, mask = to_dense_batch(z_graph, set_batch)
        z_n = self.ln1(z_dense)
        attn, _ = self.mha(z_n, z_n, z_n, key_padding_mask=~mask, need_weights=False)
        z_dense = z_dense + attn
        z_dense = z_dense + self.ffn(self.ln2(z_dense))
        S = z_dense.shape[0]
        seed = self.seed.expand(S, -1, -1)
        zn = self.ln3(z_dense)
        z_set, _ = self.pma(seed, zn, zn, key_padding_mask=~mask, need_weights=False)
        return z_set.squeeze(1)


class PipelineModel(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, set_head, num_heads, dropout=0.1):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden, num_layers, dropout=dropout)
        if set_head == "deepsets":
            self.set_head = DeepSetsHead(hidden, dropout=dropout)
        elif set_head == "settransformer":
            self.set_head = SetTransformerHead(hidden, num_heads, dropout=dropout)
        else:
            raise ValueError(set_head)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            make_activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch, set_batch):
        h = self.encoder(x, edge_index, batch)
        z_graph = global_mean_pool(h, batch)
        z_set = self.set_head(z_graph, set_batch)
        graph_at_node = z_graph[batch]
        set_at_node = z_set[set_batch[batch]]
        feat = torch.cat([h, graph_at_node, set_at_node], dim=-1)
        return self.head(feat).squeeze(-1)


class GraphSetConvModel(nn.Module):
    def __init__(
        self, in_dim, hidden, num_layers, num_heads, node_set_mode, dropout=0.1
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
                    ffn_multiplier=4,
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

    def forward(self, x, edge_index, batch, set_batch):
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x, edge_index, batch, set_batch)
        return self.head(x).squeeze(-1)


MODEL_KEYS = {
    "gsc-ca": "GraphSetConv-CrossAttn",
    "gsc-bc": "GraphSetConv-Broadcast",
    "gcn-ds": "GCN+DeepSets",
    "gcn-st": "GCN+SetTransformer",
}


def build_model(key, in_dim, hidden, num_layers, num_heads, dropout=0.1):
    if key == "gsc-ca":
        return GraphSetConvModel(
            in_dim, hidden, num_layers, num_heads, "cross_attn", dropout=dropout
        )
    if key == "gsc-bc":
        return GraphSetConvModel(
            in_dim, hidden, num_layers, num_heads, "broadcast", dropout=dropout
        )
    if key == "gcn-ds":
        return PipelineModel(
            in_dim, hidden, num_layers, "deepsets", num_heads, dropout=dropout
        )
    if key == "gcn-st":
        return PipelineModel(
            in_dim, hidden, num_layers, "settransformer", num_heads, dropout=dropout
        )
    raise ValueError(key)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Parameter equalisation (same scheme as synth_benchmark.py)
# ============================================================================


def find_hidden_for_target_params(builder, target, num_heads, lo=16, hi=512):
    best_h, best_p, best_err = None, None, float("inf")
    candidates = [h for h in range(lo, hi + 1) if h % num_heads == 0]
    for h in candidates:
        m = builder(h)
        p = count_params(m)
        del m
        err = abs(p - target)
        if err < best_err:
            best_err, best_h, best_p = err, h, p
        if p > target * 4:
            break
    return best_h, best_p


def equalize_models(in_dim, hidden, num_layers, num_heads, mode):
    base_hiddens = {k: hidden for k in MODEL_KEYS}
    base_params = {
        k: count_params(build_model(k, in_dim, hidden, num_layers, num_heads))
        for k in MODEL_KEYS
    }
    if mode == "none":
        return base_hiddens, base_params

    def builder_for(key):
        return lambda h: build_model(key, in_dim, h, num_layers, num_heads)

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

    raise ValueError(mode)


# ============================================================================
# Statistics (same helpers as synth_benchmark.py)
# ============================================================================


def t_ci_95(values: Sequence[float]) -> Tuple[float, float]:
    a = np.array([v for v in values if not math.isnan(v)], dtype=np.float64)
    n = a.size
    if n < 2:
        return (float(a.mean()) if n else float("nan"), float("nan"))
    mean = a.mean()
    se = a.std(ddof=1) / math.sqrt(n)
    from scipy.stats import t as t_dist

    crit = float(t_dist.ppf(0.975, df=n - 1))
    return float(mean), float(crit * se)


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


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05):
    items = [(k, p) for k, p in pvalues.items() if not math.isnan(p)]
    items.sort(key=lambda kp: kp[1])
    m = len(items)
    out = {}
    prev_adj = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        adj = max(adj, prev_adj)
        prev_adj = adj
        out[k] = (adj, adj < alpha)
    for k, p in pvalues.items():
        if math.isnan(p):
            out[k] = (float("nan"), False)
    return out


# ============================================================================
# Train / evaluate one (model, seed)
# ============================================================================


def iterate_batches(
    examples: List[ReactionExample],
    batch_size: int,
    rng: random.Random,
    shuffle: bool,
):
    """Yield lists of `batch_size` examples until exhausted (one epoch)."""
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        yield [examples[i] for i in chunk]


def evaluate(model, examples, batch_size, device) -> float:
    model.eval()
    ys, scores = [], []
    rng = random.Random(0)
    with torch.no_grad():
        for batch in iterate_batches(examples, batch_size, rng, shuffle=False):
            bb = collate(batch, device)
            logits = model(bb.x, bb.edge_index, bb.batch, bb.set_batch)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(bb.y.cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(scores)
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def train_one(
    model: nn.Module,
    train_set: List[ReactionExample],
    val_set: List[ReactionExample],
    test_set: List[ReactionExample],
    args,
    seed: int,
    device: str,
) -> Tuple[float, float, Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_rng = random.Random(seed * 7919 + 1)

    # Loss reweighting: positives are sparse (~5% by atom); a uniform BCE
    # makes the model collapse to predicting all-negative. We use a positive
    # class weight inferred from a small training-set audit.
    pos, neg = 0, 0
    for ex in train_set[: min(len(train_set), 1000)]:
        for lbl in ex.labels:
            pos += int(lbl.sum())
            neg += int((lbl == 0).sum())
    pos_weight = torch.tensor([max(neg / max(pos, 1), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = {"epoch": [], "train_loss": [], "val_ap": []}
    best_val = -float("inf")
    best_state = None
    t_start = time.time()

    n_train_batches = max(1, len(train_set) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * n_train_batches
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        nb = 0
        for batch in iterate_batches(
            train_set, args.batch_size, train_rng, shuffle=True
        ):
            bb = collate(batch, device)
            logits = model(bb.x, bb.edge_index, bb.batch, bb.set_batch)
            loss = loss_fn(logits, bb.y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss.item())
            nb += 1
        train_loss = running / max(nb, 1)
        val_ap = evaluate(model, val_set, args.batch_size, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_ap"].append(val_ap)
        if not math.isnan(val_ap) and val_ap > best_val:
            best_val = val_ap
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_ap = evaluate(model, test_set, args.batch_size, device)
    return test_ap, time.time() - t_start, history


# ============================================================================
# Output formatters (same shape as synth_benchmark.py's)
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


def write_markdown(path: Path, summary, args, equal_info, audit):
    out = ["# USPTO-15K reaction-centre prediction", ""]
    out += [
        f"- Run: `{args.out_dir}`",
        f"- Seeds: {args.seeds}",
        f"- Epochs: {args.epochs}",
        f"- Equalisation mode: `{args.equalize_params}`",
        "",
    ]
    out += ["## Dataset audit", ""]
    out += [
        "| split | reactions | mean #reactants/rxn | mean #atoms/rxn | "
        "positive density (atoms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, a in audit.items():
        out.append(
            f"| {split} | {a['n_reactions']} | {a['mean_reactants']:.2f} | "
            f"{a['mean_atoms']:.1f} | {a['pos_density']:.4f} |"
        )
    out.append("")

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

    out += ["## Test AP (mean ± 95% Student-t CI half-width over seeds)", ""]
    out += ["| model | AP | n |", "|---|---:|---:|"]
    for k in MODEL_KEYS:
        if k not in summary["per_model"]:
            continue
        pm = summary["per_model"][k]
        if pm["n"] == 0:
            out.append(f"| {MODEL_KEYS[k]} | n/a | 0 |")
        else:
            out.append(
                f"| {MODEL_KEYS[k]} | {pm['mean']:.4f} ± {pm['ci95_half']:.4f} | {pm['n']} |"
            )
    out.append("")

    out += ["## Pairwise Wilcoxon signed-rank (Holm-Bonferroni adjusted)", ""]
    out += [
        "| comparison | p (adj) | Cohen's d_z | reject H0 |",
        "|---|---:|---:|:---:|",
    ]
    for label, dat in summary["pairwise"].items():
        mark = "yes" if dat["reject"] else "no"
        p_str = "n/a" if math.isnan(dat["p_adj"]) else f"{dat['p_adj']:.3g}"
        d_str = "n/a" if math.isnan(dat["d"]) else f"{dat['d']:+.2f}"
        out.append(f"| {label} | {p_str} | {d_str} | {mark} |")
    path.write_text("\n".join(out) + "\n")


def _latex_escape(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("#", r"\#")


def write_latex(path: Path, summary, args, equal_info):
    lines = [
        r"% Auto-generated by uspto15k_benchmark.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{USPTO-15K reaction-centre prediction (per-atom AP $\uparrow$). "
        r"Mean $\pm$ 95\% Student-t CI half-width over "
        + str(args.seeds)
        + r" seeds; \textbf{bold} = best.}",
        r"\label{tab:uspto15k}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & AP & Parameters \\",
        r"\midrule",
    ]
    means = {
        k: summary["per_model"][k]["mean"]
        for k in MODEL_KEYS
        if k in summary["per_model"] and summary["per_model"][k]["n"] > 0
    }
    best = max(means.values()) if means else float("nan")
    hiddens, params = equal_info if equal_info else ({}, {})
    for k in MODEL_KEYS:
        if k not in summary["per_model"] or summary["per_model"][k]["n"] == 0:
            continue
        pm = summary["per_model"][k]
        m, h = pm["mean"], pm["ci95_half"]
        s = f"{m:.3f}\\,$\\pm$\\,{h:.3f}"
        if abs(m - best) < 1e-9:
            s = r"\textbf{" + s + "}"
        p = params.get(k, pm.get("params", 0))
        lines.append(f"{_latex_escape(MODEL_KEYS[k])} & {s} & {p:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


# ============================================================================
# Main
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing train.txt, valid.txt, test.txt "
        "(USPTO-15K release from Jin et al.).",
    )
    p.add_argument("--out-dir", default="results/uspto15k_run")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 1 seed, 2 epochs, small subsets.",
    )

    # architecture
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)

    # data subsetting (mostly for --quick or for fast iteration)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)

    # model selection
    p.add_argument(
        "--equalize-params", choices=["none", "smaller", "larger"], default="none"
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["gsc-ca", "gsc-bc", "gcn-ds", "gcn-st"],
        choices=list(MODEL_KEYS.keys()),
    )

    args = p.parse_args()
    if args.quick:
        args.epochs = 2
        args.seeds = 1
        args.max_train = 1500
        args.max_val = 200
        args.max_test = 200
    return args


def audit_split(name: str, examples: List[ReactionExample]) -> Dict:
    n_atoms_total, n_pos_total, n_react_total = 0, 0, 0
    for ex in examples:
        n_react_total += len(ex.mols)
        for lbl in ex.labels:
            n_atoms_total += lbl.shape[0]
            n_pos_total += int(lbl.sum())
    return {
        "n_reactions": len(examples),
        "mean_reactants": n_react_total / max(len(examples), 1),
        "mean_atoms": n_atoms_total / max(len(examples), 1),
        "pos_density": n_pos_total / max(n_atoms_total, 1),
    }


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    in_dim = atom_feature_dim()

    print(f"=== uspto15k_benchmark.py ===")
    print(f"out-dir = {out_dir}")
    print(f"device  = {device}")
    print(f"models  = {[MODEL_KEYS[k] for k in args.models]}")
    print(f"seeds   = {args.seeds}")
    print(f"epochs  = {args.epochs}")
    print(f"in_dim  = {in_dim}")

    # Data
    data_dir = Path(args.data_dir)
    print(f"\nLoading USPTO-15K splits from {data_dir} ...")
    train_set = load_split(data_dir / "train.txt", args.max_train)
    val_set = load_split(data_dir / "valid.txt", args.max_val)
    test_set = load_split(data_dir / "test.txt", args.max_test)

    audit = {
        "train": audit_split("train", train_set),
        "valid": audit_split("valid", val_set),
        "test": audit_split("test", test_set),
    }
    print("\n--- Split audit ---")
    for s, a in audit.items():
        print(
            f"  {s:<6} n_rxn={a['n_reactions']:>5}  "
            f"mean_reactants={a['mean_reactants']:.2f}  "
            f"mean_atoms={a['mean_atoms']:.1f}  "
            f"pos_density={a['pos_density']:.4f}"
        )

    # Param budgets
    hiddens, params = equalize_models(
        in_dim, args.hidden, args.layers, args.num_heads, args.equalize_params
    )
    print(f"\n[equalize-params={args.equalize_params}]")
    for k in MODEL_KEYS:
        print(f"  {MODEL_KEYS[k]:<28} hidden={hiddens[k]:>4}  params={params[k]:>8,}")

    # Persist config
    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "model_param_counts": params,
            "model_hidden_dims": hiddens,
            "audit": audit,
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

    # Run
    raw_results = {k: [] for k in args.models}
    timings = {k: [] for k in args.models}
    history_records = {k: {} for k in args.models}
    flat_csv: List[Dict] = []

    for mkey in args.models:
        display = MODEL_KEYS[mkey]
        print(f"\n========== {display} ==========")
        for sidx in range(args.seeds):
            seed = args.seed_offset + sidx
            model = build_model(
                mkey,
                in_dim,
                hiddens[mkey],
                args.layers,
                args.num_heads,
                dropout=args.dropout,
            ).to(device)
            p_count = count_params(model)
            score, elapsed, hist = train_one(
                model, train_set, val_set, test_set, args, seed, device
            )
            raw_results[mkey].append(score)
            timings[mkey].append(elapsed)
            history_records[mkey][seed] = hist
            print(
                f"  seed={seed:>2}  test_AP={score:.4f}  "
                f"({elapsed:.1f}s, {p_count:,} params)"
            )
            flat_csv.append(
                {
                    "model_key": mkey,
                    "model_name": display,
                    "seed": seed,
                    "hidden": hiddens[mkey],
                    "params": p_count,
                    "test_ap": score,
                    "wall_seconds": round(elapsed, 2),
                }
            )
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Aggregate
    per_model = {}
    for mkey in MODEL_KEYS:
        scores = raw_results.get(mkey, [])
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

    active = [k for k in args.models if per_model[k]["n"] >= 2]
    pairwise_raw = {}
    for k1, k2 in combinations(active, 2):
        a = per_model[k1]["scores"]
        b = per_model[k2]["scores"]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        p = wilcoxon_signed_rank(a, b)
        d = cohens_d_paired(a, b)
        pairwise_raw[f"{MODEL_KEYS[k1]} vs {MODEL_KEYS[k2]}"] = (p, d)
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

    summary = {
        "per_model": per_model,
        "pairwise": pairwise,
        "wall_seconds": {
            k: float(np.mean(v)) if v else float("nan") for k, v in timings.items()
        },
    }

    # Outputs
    write_json(
        out_dir / "raw.json",
        {
            "summary": summary,
            "raw_scores": raw_results,
            "timings": timings,
            "history": history_records,
            "audit": audit,
        },
    )
    write_csv(out_dir / "summary.csv", flat_csv)
    write_markdown(out_dir / "summary.md", summary, args, (hiddens, params), audit)
    write_latex(out_dir / "summary.tex", summary, args, (hiddens, params))

    # Console summary
    print(f"\n========== summary ==========\n")
    for mkey in args.models:
        pm = per_model[mkey]
        if pm["n"] == 0:
            continue
        print(
            f"  {MODEL_KEYS[mkey]:<28} "
            f"AP={pm['mean']:.4f} ± {pm['ci95_half']:.4f}  "
            f"(n={pm['n']}, params={pm['params']:,})"
        )
    if pairwise:
        print("\n  pairwise (Holm-adjusted):")
        for label, dat in pairwise.items():
            tag = "*" if dat["reject"] else " "
            p_str = "n/a" if math.isnan(dat["p_adj"]) else f"{dat['p_adj']:.3g}"
            d_str = "n/a" if math.isnan(dat["d"]) else f"{dat['d']:+.2f}"
            print(f"   {tag} {label}:  p_adj={p_str},  d_z={d_str}")
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    run(parse_args())
