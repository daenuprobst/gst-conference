"""
atom_mapping_benchmark.py
=========================

Atom-mapping benchmark on USPTO-50K (Schneider et al. 2016
classification subset of Lowe's USPTO grants extraction; Dai et al.
2019 cleaning + predefined splits, ~50K reactions in 10 reaction
classes; or the LocalMapper 2024 remap with cleaner mappings).
Per-atom prediction in a set context: given the reactants and
products as molecule sets, predict the bijection between product
atoms and reactant atoms.

The architectural test
----------------------
We evaluate four set aggregators:
    gcn-ds      GCN encoder + per-element DeepSets (mol-level set context)
    gcn-st      GCN encoder + per-element SetTransformer (mol-level)
    gsc-bc      GraphSetConv (broadcast set context, atom-level interleaved)
    gsc-ca      GraphSetConv (per-node cross-attention, atom-level interleaved)

Set framing
-----------
By default the encoder sees only within-side molecules:
    mol_to_set = (rxn * 2) + side       ("siamese" framing)
which is the cleanest baseline since the assignment head does ALL
the cross-side work, and the encoder comparison is purely about
how each architecture refines per-atom representations under
within-side set context.

Earlier exploratory runs included a "joint" framing (mol_to_set =
rxn, encoder sees both sides) to test whether GSC's interleaved
cross-side attention extracts more from encoder-level cross-side
context than pipelines do. Across two datasets (Golden, USPTO-50K)
the siamese-vs-joint deltas were uniformly within noise of zero
for all four architectures, so we fix the framing to siamese for
the main results. The --joint flag flips to joint for ablation.

Why pipelines use PER-ELEMENT set heads
---------------------------------------
For atom-level tasks, a single broadcast vector per set would
handicap pipelines unfairly. The per-element variants return one
contextualized vector per molecule; that vector is then broadcast to
the atoms of THAT molecule. This is the right pipeline analog of
GSC's per-block attention.

Side identity
-------------
Atom mapping is asymmetric (R <-> P). A learned side embedding is
added at the atom level by default. --no-side-identity disables it.

Assignment head
---------------
After encoding, atoms are split by side. The head computes:
    S[p, r] = (proj_p(z_p) . proj_r(z_r)) / sqrt(D)
masked to within-reaction pairs. Loss is per-product-atom
cross-entropy against the ground-truth reactant index.

Metrics
-------
- atom_acc       fraction of mapped product atoms correctly assigned
- reaction_acc   fraction of reactions where ALL mapped product atoms
                 are correctly assigned (chemist-relevant number)

Splits
------
The standard Schneider/Dai splits are PREDEFINED (separate train,
val, test CSV files). We use them as-is. Random re-shuffling over
USPTO-50K leaks (reactions from the same patent share intermediates),
so do NOT re-split.

Data setup
----------
Download one of:

  (1) Standard GLN/Schneider raw split:
        https://figshare.com/articles/dataset/USPTO-50K_raw_/25459573
        (or via https://github.com/Hanjun-Dai/GLN dropbox link)

  (2) LocalMapper-remapped variant (recommended, cleaner mappings):
        https://figshare.com/articles/dataset/USPTO_reaction_datasets_remapped_by_LocalMapper/25046471

Place raw_train.csv, raw_val.csv, raw_test.csv into a directory and
pass it as --data-dir. (If your distribution names them
train.csv/val.csv/test.csv, override with --train-file etc.)

Outputs (per --out-dir/):
    raw.json                    per-seed scores, history, per-rxn errs
    summary.csv                 flat re-analyzable table
    summary.md                  human-readable report
    summary.tex                 booktabs LaTeX (main results)
    error_distribution.csv      per-model bucket counts + concentration
    error_distribution.md       human-readable error-distribution report
    error_distribution.tex      booktabs LaTeX (error distribution)
    config.json                 exact run config + env

Usage
-----
    # Smoke test (1 seed, sub-sampled, small model)
    python atom_mapping_benchmark.py --quick \\
        --data-dir data/uspto50k

    # Full run
    python atom_mapping_benchmark.py \\
        --data-dir data/uspto50k \\
        --models gcn-ds gcn-st gsc-bc gsc-ca \\
        --seeds 5 \\
        --out-dir results/uspto50k_full
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uspto50k_data import (
    ATOM_FEAT_DIM,
    AtomMappingExample,
    load_uspto50k,
)

from graph_set_transformer.models.gst import GraphSetConv


# =============================================================================
# Pure-torch pool helpers (no torch_scatter dependency)
# =============================================================================


def global_add_pool_safe(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_groups = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = x.new_zeros(n_groups, x.size(-1))
    out.index_add_(0, batch, x)
    return out


def global_mean_pool_safe(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_groups = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = x.new_zeros(n_groups, x.size(-1))
    out.index_add_(0, batch, x)
    counts = x.new_zeros(n_groups)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(-1)


# =============================================================================
# Per-element set heads (the key change from olfaction)
# =============================================================================


class DeepSetsHeadPerElement(nn.Module):
    """Per-element DeepSets:
        z_i' = rho(phi(z_i) + sum_j phi(z_j))
    Each element receives its set's pooled context as an additive
    update (rather than the canonical pooling-only output)."""

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

    def forward(self, z: torch.Tensor, set_batch: torch.Tensor) -> torch.Tensor:
        h = self.phi(z)
        z_set = global_add_pool_safe(h, set_batch)
        return self.rho(h + z_set[set_batch])


class SetTransformerHeadPerElement(nn.Module):
    """Per-element SetTransformer: a single transformer encoder block
    over set elements (self-attention + FFN), no PMA seed pooling.
    Returns one contextualized vector per element."""

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

    def forward(self, z: torch.Tensor, set_batch: torch.Tensor) -> torch.Tensor:
        from torch_geometric.utils import to_dense_batch

        z_dense, mask = to_dense_batch(z, set_batch)
        zn = self.ln1(z_dense)
        attn, _ = self.mha(zn, zn, zn, key_padding_mask=~mask, need_weights=False)
        z_dense = z_dense + attn
        z_dense = z_dense + self.ffn(self.ln2(z_dense))
        # Flatten back to [N, D]; to_dense_batch preserves input order,
        # so this returns elements in the same order as `z`.
        return z_dense[mask]


# =============================================================================
# GCN encoder for pipeline baselines
# =============================================================================


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


def _make_attn_pool(hidden: int) -> nn.Module:
    """Atom -> molecule AttentionalAggregation (shared by pipelines)."""
    from torch_geometric.nn import AttentionalAggregation

    return AttentionalAggregation(
        gate_nn=nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        ),
        nn=None,
    )


# =============================================================================
# Model registry
# =============================================================================


MODEL_KEYS = {
    "gsc-bc": "GraphSetConv-Broadcast",
    "gsc-ca": "GraphSetConv-CrossAttn",
    "gcn-ds": "GCN+DeepSets",
    "gcn-st": "GCN+SetTransformer",
}

PIPELINE_KEYS = {"gcn-ds", "gcn-st"}
GSC_KEYS = {"gsc-bc", "gsc-ca"}

FRAMING_KEYS = ("siamese", "joint")  # for the model's set-keying flag


# =============================================================================
# AtomMappingModel
# =============================================================================


class AtomMappingModel(nn.Module):
    """Unified Siamese / Joint atom-mapping model.

    Inputs (via PackedAtomBatch):
        x:            atom features for all atoms in the batch
        edge_index:   bonds (concatenated, with offsets)
        atom_to_mol:  atom -> molecule id (0..G-1)
        mol_to_rxn:   molecule -> reaction id (0..R-1)
        mol_to_side:  molecule -> 0 (reactant) or 1 (product)

    Pipeline (gcn-ds / gcn-st):
        x -> input_proj (+ side_emb)
        x -> GCNStack         (atom-level message passing)
        z_atom -> node_pool   (atom -> mol)
        z_mol  -> set_head    (per-mol set context, one vector per mol)
        z_atom_final = z_atom + z_mol_ctx[atom_to_mol]   (broadcast)

    GSC (gsc-bc / gsc-ca):
        x -> input_proj (+ side_emb)
        for blk in blocks:
            x = blk(x, edge_index, atom_to_mol, mol_to_set)
        z_atom_final = x

    Set keying:
        siamese:  mol_to_set = mol_to_rxn * 2 + mol_to_side
        joint:    mol_to_set = mol_to_rxn

    Assignment head:
        Split z_atom_final by side -> z_r [N_r, D], z_p [N_p, D].
        S[p, r] = (proj_p(z_p) . proj_r(z_r)) / sqrt(D)
        S masked to within-reaction pairs (-inf elsewhere).
        Returns S; loss/accuracy computed in the training loop.
    """

    def __init__(
        self,
        model_key: str,
        framing: str,
        num_layers: int,
        hidden: int,
        num_heads: int,
        dropout: float,
        atom_feat_dim: int = ATOM_FEAT_DIM,
        use_side_identity: bool = True,
    ):
        super().__init__()
        if framing not in FRAMING_KEYS:
            raise ValueError(f"unknown framing: {framing!r}")
        self.model_key = model_key
        self.framing = framing
        self.is_pipeline = model_key in PIPELINE_KEYS
        self.use_side_identity = use_side_identity

        self.input_proj = nn.Linear(atom_feat_dim, hidden)

        if use_side_identity:
            self.side_emb = nn.Embedding(2, hidden)
            nn.init.normal_(self.side_emb.weight, std=0.02)
        else:
            self.side_emb = None

        if self.is_pipeline:
            self.encoder = GCNStack(hidden, hidden, num_layers, dropout=dropout)
            self.node_pool = _make_attn_pool(hidden)
            if model_key == "gcn-ds":
                self.set_head = DeepSetsHeadPerElement(hidden, dropout=dropout)
            elif model_key == "gcn-st":
                self.set_head = SetTransformerHeadPerElement(
                    hidden, num_heads, dropout=dropout
                )
            else:
                raise ValueError(f"unknown pipeline: {model_key!r}")
        else:
            blocks = []
            for _ in range(num_layers):
                if model_key == "gsc-bc":
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
                elif model_key == "gsc-ca":
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
                    raise ValueError(f"unknown gsc variant: {model_key!r}")
            self.blocks = nn.ModuleList(blocks)

        # Assignment head: separate projections for reactant and product
        # atoms (the task is asymmetric).
        self.proj_r = nn.Linear(hidden, hidden)
        self.proj_p = nn.Linear(hidden, hidden)

    def forward(self, batch_obj):
        x_atom = batch_obj.x
        edge_index = batch_obj.edge_index
        atom_to_mol = batch_obj.atom_to_mol
        mol_to_rxn = batch_obj.mol_to_rxn
        mol_to_side = batch_obj.mol_to_side

        # Per-atom side and reaction (gather from per-mol indices).
        atom_to_side = mol_to_side[atom_to_mol]
        atom_to_rxn = mol_to_rxn[atom_to_mol]

        # Input projection + optional side embedding.
        x = self.input_proj(x_atom)
        if self.side_emb is not None:
            x = x + self.side_emb(atom_to_side)

        # Set keying per framing.
        if self.framing == "siamese":
            mol_to_set = mol_to_rxn * 2 + mol_to_side
        else:
            mol_to_set = mol_to_rxn

        # Encoder.
        if self.is_pipeline:
            x = self.encoder(x, edge_index)
            z_mol = self.node_pool(x, atom_to_mol)
            z_mol_ctx = self.set_head(z_mol, mol_to_set)
            # Broadcast per-mol set context back to atoms of that mol.
            z_atom = x + z_mol_ctx[atom_to_mol]
        else:
            for blk in self.blocks:
                x = blk(x, edge_index, atom_to_mol, mol_to_set)
            z_atom = x

        # Split by side.
        mask_r = atom_to_side == 0
        mask_p = atom_to_side == 1
        z_r = z_atom[mask_r]
        z_p = z_atom[mask_p]
        rxn_r = atom_to_rxn[mask_r]
        rxn_p = atom_to_rxn[mask_p]

        # Affinity matrix [N_p, N_r], masked to within-reaction.
        q = self.proj_p(z_p)
        k = self.proj_r(z_r)
        S = (q @ k.t()) / math.sqrt(q.size(-1))
        valid = rxn_p.unsqueeze(-1) == rxn_r.unsqueeze(0)
        S = S.masked_fill(~valid, float("-inf"))
        return S


def build_model(
    key: str,
    framing: str,
    num_layers: int,
    hidden: int,
    num_heads: int,
    dropout: float,
    atom_feat_dim: int = ATOM_FEAT_DIM,
    use_side_identity: bool = True,
) -> nn.Module:
    return AtomMappingModel(
        model_key=key,
        framing=framing,
        num_layers=num_layers,
        hidden=hidden,
        num_heads=num_heads,
        dropout=dropout,
        atom_feat_dim=atom_feat_dim,
        use_side_identity=use_side_identity,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        m = builder(h)
        p = count_params(m)
        del m
        err = abs(p - target)
        if err < best_err:
            best_err, best_h, best_p = err, h, p
        if best_p is not None and p > target * 4:
            break
    return best_h, best_p


def equalize_models(
    active_keys: List[str],
    args,
    feat_dim: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Match parameter counts across architectures (with the
    --equalize-params strategy)."""
    base_hidden = args.hidden
    framing = args.framing

    def build_for(key: str, hidden: int) -> nn.Module:
        return build_model(
            key,
            framing,
            args.layers,
            hidden,
            args.num_heads,
            args.dropout,
            atom_feat_dim=feat_dim,
            use_side_identity=getattr(args, "use_side_identity", True),
        )

    base_hiddens = {k: base_hidden for k in active_keys}
    base_params = {k: count_params(build_for(k, base_hidden)) for k in active_keys}

    if args.equalize_params == "none":
        return base_hiddens, base_params

    pipeline_keys = [k for k in active_keys if k in PIPELINE_KEYS]
    gsc_keys = [k for k in active_keys if k in GSC_KEYS]

    if args.equalize_params == "smaller":
        if not pipeline_keys:
            return base_hiddens, base_params
        target_key = min(pipeline_keys, key=lambda k: base_params[k])
        target_p = base_params[target_key]
        hiddens = {k: base_hiddens[k] for k in pipeline_keys}
        params = {k: base_params[k] for k in pipeline_keys}
        for k in gsc_keys:
            h, p = find_hidden_for_target_params(
                lambda hidden, kk=k: build_for(kk, hidden),
                target_p,
                args.num_heads,
            )
            hiddens[k] = h
            params[k] = p
        return hiddens, params

    if args.equalize_params == "larger":
        if not gsc_keys:
            return base_hiddens, base_params
        target_p = max(base_params[k] for k in gsc_keys)
        hiddens = {k: base_hiddens[k] for k in gsc_keys}
        params = {k: base_params[k] for k in gsc_keys}
        for k in pipeline_keys:
            h, p = find_hidden_for_target_params(
                lambda hidden, kk=k: build_for(kk, hidden),
                target_p,
                args.num_heads,
            )
            hiddens[k] = h
            params[k] = p
        return hiddens, params

    raise ValueError(f"Unknown equalize_params: {args.equalize_params}")


# =============================================================================
# Batch packing
# =============================================================================


class PackedAtomBatch:
    """One mini-batch of atom-mapping reactions packed into flat tensors.

    Conventions (S = batch size in reactions, G = total mols, N = atoms):
        x:                       [N, ATOM_FEAT_DIM]
        edge_index:              [2, E]
        atom_to_mol:             [N]    atom -> molecule id (0..G-1)
        mol_to_rxn:              [G]    molecule -> reaction id (0..S-1)
        mol_to_side:             [G]    0 (reactant) or 1 (product)
        target_global_r_idx:     [N_p]  for each PRODUCT atom, the index
                                        of its reactant counterpart in
                                        the BATCH'S all-reactant-atoms
                                        subset (i.e. valid range
                                        [0, N_r_total)), or -1 if
                                        unmapped.
        product_rxn_idx:         [N_p]  reaction id per product atom
                                        (used for reaction-level acc).

    Reactant and product atoms are interleaved per-reaction in
    construction order: rxn0_reactant_atoms, rxn0_product_atoms,
    rxn1_reactant_atoms, ... This means selecting atom_to_side==0
    yields a contiguous chunk per reaction, so global reactant
    indices are well-defined as
        offset[rxn] + local_reactant_idx_within_rxn
    where offset[rxn] = sum of N_r over earlier reactions in batch.
    """

    def __init__(
        self,
        x,
        edge_index,
        atom_to_mol,
        mol_to_rxn,
        mol_to_side,
        target_global_r_idx,
        product_rxn_idx,
        rxn_ids,
    ):
        self.x = x
        self.edge_index = edge_index
        self.atom_to_mol = atom_to_mol
        self.mol_to_rxn = mol_to_rxn
        self.mol_to_side = mol_to_side
        self.target_global_r_idx = target_global_r_idx
        self.product_rxn_idx = product_rxn_idx
        self.rxn_ids = rxn_ids

    def to(self, device):
        for attr in (
            "x",
            "edge_index",
            "atom_to_mol",
            "mol_to_rxn",
            "mol_to_side",
            "target_global_r_idx",
            "product_rxn_idx",
        ):
            setattr(self, attr, getattr(self, attr).to(device))
        return self


def collate(samples: List[AtomMappingExample]) -> PackedAtomBatch:
    xs: List[torch.Tensor] = []
    edges: List[torch.Tensor] = []
    atom_to_mol_chunks: List[torch.Tensor] = []
    mol_to_rxn_chunks: List[torch.Tensor] = []
    mol_to_side_chunks: List[torch.Tensor] = []
    target_global: List[int] = []
    product_rxn_idx: List[int] = []
    rxn_ids: List[str] = []

    g_off = 0  # running count of molecules
    n_off = 0  # running count of atoms (for edge_index offset)
    cum_reactant_atoms = 0  # sum of reactant atoms over earlier reactions

    for r_idx, ex in enumerate(samples):
        rxn_ids.append(ex.rxn_id)
        n_r_atoms = ex.num_reactant_atoms
        # Reactants first (side=0).
        for g in ex.reactant_mols:
            xs.append(g.x)
            if g.edge_index.numel() > 0:
                edges.append(g.edge_index + n_off)
            atom_to_mol_chunks.append(
                torch.full((g.num_nodes,), g_off, dtype=torch.long)
            )
            mol_to_rxn_chunks.append(torch.tensor([r_idx], dtype=torch.long))
            mol_to_side_chunks.append(torch.tensor([0], dtype=torch.long))
            n_off += g.num_nodes
            g_off += 1
        # Products (side=1).
        for g in ex.product_mols:
            xs.append(g.x)
            if g.edge_index.numel() > 0:
                edges.append(g.edge_index + n_off)
            atom_to_mol_chunks.append(
                torch.full((g.num_nodes,), g_off, dtype=torch.long)
            )
            mol_to_rxn_chunks.append(torch.tensor([r_idx], dtype=torch.long))
            mol_to_side_chunks.append(torch.tensor([1], dtype=torch.long))
            n_off += g.num_nodes
            g_off += 1
        # Targets for this reaction's product atoms (in flat product order).
        for tgt_local in ex.product_to_reactant:
            if tgt_local < 0:
                target_global.append(-1)
            else:
                target_global.append(cum_reactant_atoms + tgt_local)
            product_rxn_idx.append(r_idx)
        cum_reactant_atoms += n_r_atoms

    edge_index = (
        torch.cat(edges, dim=1) if edges else torch.zeros((2, 0), dtype=torch.long)
    )
    return PackedAtomBatch(
        x=torch.cat(xs, dim=0),
        edge_index=edge_index,
        atom_to_mol=torch.cat(atom_to_mol_chunks, dim=0),
        mol_to_rxn=torch.cat(mol_to_rxn_chunks, dim=0),
        mol_to_side=torch.cat(mol_to_side_chunks, dim=0),
        target_global_r_idx=torch.tensor(target_global, dtype=torch.long),
        product_rxn_idx=torch.tensor(product_rxn_idx, dtype=torch.long),
        rxn_ids=rxn_ids,
    )


# =============================================================================
# Statistics
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


# Error-count buckets used for histogram reporting. Buckets are
# (lo_inclusive, hi_inclusive). Use math.inf for the open top bucket.
ERROR_BUCKETS: List[Tuple[int, float]] = [
    (0, 0),
    (1, 1),
    (2, 3),
    (4, 7),
    (8, math.inf),
]
ERROR_BUCKET_LABELS = ["0", "1", "2-3", "4-7", "8+"]


def _bucket_idx(n_err: int) -> int:
    for i, (lo, hi) in enumerate(ERROR_BUCKETS):
        if lo <= n_err <= hi:
            return i
    return len(ERROR_BUCKETS) - 1  # numerical safety; should never hit


def analyze_error_distribution(
    per_rxn_pooled: List[Tuple[int, int]],
) -> Dict:
    """Summarize a list of (n_atoms, n_errors) per reaction.

    Returns a dict with:
      - n_rxns                         total reactions
      - n_atoms_total                  total mapped product atoms
      - atom_acc                       1 - (errors/atoms), recomputed
      - reaction_acc_observed          empirical
      - reaction_acc_iid               P(all-correct) under per-atom
                                       independence at the empirical
                                       atom error rate p — the
                                       counterfactual we expect if
                                       errors had no within-rxn
                                       structure
      - concentration_ratio            observed / iid
                                       >1 = errors more concentrated
                                            in fewer reactions
                                            (good for reaction_acc)
                                       <1 = errors more scattered
                                            (bad for reaction_acc)
      - bucket_counts                  raw counts per error bucket
      - bucket_fracs                   fractions per error bucket
      - mean_errors_per_failed_rxn     average # errors among reactions
                                       that have any errors
                                       (the size of "failed" reactions)
    """
    if not per_rxn_pooled:
        return {
            "n_rxns": 0,
            "n_atoms_total": 0,
            "atom_acc": float("nan"),
            "reaction_acc_observed": float("nan"),
            "reaction_acc_iid": float("nan"),
            "concentration_ratio": float("nan"),
            "bucket_counts": [0] * len(ERROR_BUCKETS),
            "bucket_fracs": [float("nan")] * len(ERROR_BUCKETS),
            "mean_errors_per_failed_rxn": float("nan"),
        }
    n_rxns = len(per_rxn_pooled)
    n_atoms_total = sum(k for k, _ in per_rxn_pooled)
    n_errors_total = sum(e for _, e in per_rxn_pooled)
    atom_acc = 1.0 - n_errors_total / max(n_atoms_total, 1)
    p = 1.0 - atom_acc  # per-atom error rate

    # Observed reaction_acc.
    n_full = sum(1 for _, e in per_rxn_pooled if e == 0)
    reaction_acc_observed = n_full / n_rxns

    # i.i.d. counterfactual: average over reactions of (1-p)^k.
    if n_rxns:
        reaction_acc_iid = float(np.mean([(1.0 - p) ** k for k, _ in per_rxn_pooled]))
    else:
        reaction_acc_iid = float("nan")
    concentration_ratio = (
        reaction_acc_observed / reaction_acc_iid
        if reaction_acc_iid > 0
        else float("inf")
    )

    # Bucket histogram.
    counts = [0] * len(ERROR_BUCKETS)
    for _, e in per_rxn_pooled:
        counts[_bucket_idx(e)] += 1
    fracs = [c / n_rxns for c in counts]

    # Errors per failed reaction.
    failed_errs = [e for _, e in per_rxn_pooled if e > 0]
    mean_errs_failed = float(np.mean(failed_errs)) if failed_errs else float("nan")

    return {
        "n_rxns": n_rxns,
        "n_atoms_total": n_atoms_total,
        "atom_acc": atom_acc,
        "reaction_acc_observed": reaction_acc_observed,
        "reaction_acc_iid": reaction_acc_iid,
        "concentration_ratio": concentration_ratio,
        "bucket_counts": counts,
        "bucket_fracs": fracs,
        "mean_errors_per_failed_rxn": mean_errs_failed,
    }


# =============================================================================
# Train / eval
# =============================================================================


def iterate_minibatches(items, batch_size, rng, shuffle=True):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [items[j] for j in idx[i : i + batch_size]]


def _compute_loss_and_acc(S: torch.Tensor, batch: PackedAtomBatch):
    """Compute per-product-atom CE loss and per-atom correctness flags.
    Returns (loss, atom_correct: BoolTensor[N_p], valid_mask: BoolTensor[N_p],
    preds: LongTensor[N_p], targets: LongTensor[N_p])."""
    targets = batch.target_global_r_idx
    valid = targets >= 0
    if valid.sum() == 0:
        # No supervision in this batch (shouldn't normally happen).
        zero = S.new_zeros(())
        return zero, valid, valid, S.argmax(dim=-1), targets
    # Compute CE only over rows with valid targets.
    S_v = S[valid]
    t_v = targets[valid]
    loss = F.cross_entropy(S_v, t_v)
    preds = S.argmax(dim=-1)
    atom_correct = (preds == targets) & valid
    return loss, atom_correct, valid, preds, targets


def eval_set(model, items, args, device, return_per_rxn: bool = False):
    """Evaluate atom_acc and reaction_acc on a list of examples.

    If return_per_rxn=True, also returns a list of per-reaction
    (n_atoms, n_errors) tuples — used after the run to characterize
    error distributions across architectures."""
    if not items:
        out = {"atom_acc": float("nan"), "reaction_acc": float("nan")}
        if return_per_rxn:
            out["per_rxn"] = []
        return out
    model.eval()
    all_atom_correct: List[torch.Tensor] = []
    all_valid: List[torch.Tensor] = []
    # Per-product-atom rxn id, offset across batches so reaction ids
    # are unique batch-to-batch.
    all_rxn_idx: List[torch.Tensor] = []
    rxn_offset = 0
    with torch.no_grad():
        for samples in iterate_minibatches(
            items, args.batch_size, random.Random(0), shuffle=False
        ):
            pb = collate(samples).to(device)
            S = model(pb)
            _, atom_correct, valid, _, _ = _compute_loss_and_acc(S, pb)
            all_atom_correct.append(atom_correct.cpu())
            all_valid.append(valid.cpu())
            all_rxn_idx.append(pb.product_rxn_idx.cpu() + rxn_offset)
            rxn_offset += len(samples)
    atom_correct = torch.cat(all_atom_correct)
    valid = torch.cat(all_valid)
    rxn_idx = torch.cat(all_rxn_idx)
    n_valid = int(valid.sum().item())
    atom_acc = float(atom_correct.sum().item()) / max(n_valid, 1)
    # Reaction-level: count rxns where ALL valid atoms are correct.
    n_full = 0
    n_rxns = 0
    per_rxn: List[Tuple[int, int]] = []  # (n_mapped_atoms, n_errors)
    for r in rxn_idx.unique().tolist():
        m = (rxn_idx == r) & valid
        n_atoms = int(m.sum().item())
        if n_atoms == 0:
            continue
        n_rxns += 1
        n_corr = int((atom_correct & m).sum().item())
        if n_corr == n_atoms:
            n_full += 1
        if return_per_rxn:
            per_rxn.append((n_atoms, n_atoms - n_corr))
    reaction_acc = float(n_full) / max(n_rxns, 1)
    out = {"atom_acc": atom_acc, "reaction_acc": reaction_acc}
    if return_per_rxn:
        out["per_rxn"] = per_rxn
    return out


def _fmt_duration(seconds: float) -> str:
    if not (0 <= seconds < float("inf")):
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def train_one(model, train, val, test, args, seed, device, progress_prefix: str = ""):
    """Adam + masked CE loss; early stop on val atom_acc."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("-inf")
    best_state = None
    best_epoch = 0
    epochs_since_improvement = 0
    history = []
    t_start = time.time()
    use_val = len(val) > 0
    log_epoch = max(1, getattr(args, "log_every_epoch", 1))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        for samples in iterate_minibatches(train, args.batch_size, rng):
            pb = collate(samples).to(device)
            S = model(pb)
            loss, _, valid, _, _ = _compute_loss_and_acc(S, pb)
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
            n_atoms = int(valid.sum().item())
            running_loss += loss.item() * max(n_atoms, 1)
            running_n += max(n_atoms, 1)
        if running_n == 0:
            print(f"    [seed={seed} epoch={epoch}] no valid atoms; aborting")
            break
        train_loss = running_loss / max(running_n, 1)

        if use_val:
            v = eval_set(model, val, args, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_atom_acc": v["atom_acc"],
                    "val_reaction_acc": v["reaction_acc"],
                }
            )
            improved = math.isfinite(v["atom_acc"]) and v["atom_acc"] > best_val
            if improved:
                best_val = v["atom_acc"]
                best_epoch = epoch
                best_state = {
                    k: t.detach().clone() for k, t in model.state_dict().items()
                }
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
        else:
            history.append({"epoch": epoch, "train_loss": train_loss})
            improved = False

        force_print = (
            epoch == 1
            or improved
            or epoch == args.epochs
            or (use_val and epochs_since_improvement >= args.patience)
        )
        if force_print or (epoch % log_epoch == 0):
            tag = " *" if improved else "  "
            if use_val:
                v = history[-1]
                msg = (
                    f"{progress_prefix}"
                    f"epoch {epoch:>4d}/{args.epochs}{tag} "
                    f"train_loss={train_loss:.4f}  "
                    f"val_atom={v['val_atom_acc']:.4f}  "
                    f"val_rxn={v['val_reaction_acc']:.4f}  "
                    f"best_atom={best_val:.4f}@ep{best_epoch}  "
                    f"total={_fmt_duration(time.time() - t_start)}"
                )
            else:
                msg = (
                    f"{progress_prefix}"
                    f"epoch {epoch:>4d}/{args.epochs}{tag} "
                    f"train_loss={train_loss:.4f}  "
                    f"total={_fmt_duration(time.time() - t_start)}"
                )
            print(msg)
            sys.stdout.flush()

        if use_val and epochs_since_improvement >= args.patience:
            break

    if use_val and best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = eval_set(model, test, args, device, return_per_rxn=True)
    return test_metrics, time.time() - t_start, history


# =============================================================================
# Output writers
# =============================================================================


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def write_markdown(path: Path, summary: Dict, args):
    out = ["# USPTO-50K atom-mapping benchmark", ""]
    out += [
        f"- Models: {' '.join(args.models)}",
        f"- Framing: `{args.framing}`",
        f"- Seeds: {args.seeds}",
        f"- Epochs (max): {args.epochs}, patience: {args.patience}",
        f"- Hidden: {args.hidden}, Layers: {args.layers}",
        f"- Equalize params: `{args.equalize_params}`",
        f"- Side identity: `{'on' if args.use_side_identity else 'off'}`",
        f"- Canonicalize: `{'on' if args.canonicalize else 'off'}`",
        f"- Data: `{args.data_dir}` "
        f"(train={args.train_file}, val={args.val_file}, test={args.test_file})",
        "",
        "Metrics: atom_acc = fraction of mapped product atoms correctly",
        "assigned; reaction_acc = fraction of reactions where ALL mapped",
        "product atoms are correct.",
        "",
        "## Test metrics (mean ± 95% CI)",
        "",
        "| model | params | atom_acc | reaction_acc |",
        "|---|---:|---|---|",
    ]
    atom_means = {k: pm["atom_acc"]["mean"] for k, pm in summary["per_model"].items()}
    valid = [v for v in atom_means.values() if not math.isnan(v)]
    best_a = max(valid) if valid else float("nan")
    for mkey, pm in summary["per_model"].items():
        atom_s = (
            f"{pm['atom_acc']['mean']:.4f} ± {pm['atom_acc']['ci95']:.4f}"
            if not math.isnan(pm["atom_acc"]["mean"])
            else "n/a"
        )
        rxn_s = (
            f"{pm['reaction_acc']['mean']:.4f} ± {pm['reaction_acc']['ci95']:.4f}"
            if not math.isnan(pm["reaction_acc"]["mean"])
            else "n/a"
        )
        if (
            not math.isnan(pm["atom_acc"]["mean"])
            and abs(pm["atom_acc"]["mean"] - best_a) < 1e-9
        ):
            atom_s = f"**{atom_s}**"
        out.append(f"| {MODEL_KEYS[mkey]} | {pm['params']:,} | {atom_s} | {rxn_s} |")
    out.append("")
    path.write_text("\n".join(out) + "\n")


def _latex_escape(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("#", r"\#")


def write_latex(path: Path, summary, args):
    """Main results table."""
    lines = [
        r"% Auto-generated by atom_mapping_benchmark.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{USPTO-50K atom-mapping benchmark: per-atom and "
            r"per-reaction accuracy by architecture, mean $\pm$ 95\% "
            r"CI over "
            + str(args.seeds)
            + r" seeds. Framing: \texttt{"
            + args.framing
            + r"}. \textbf{Bold} = best atom\_acc. "
            + r"Equalize-params: \texttt{"
            + args.equalize_params
            + r"}.}"
        ),
        r"\label{tab:atom_mapping_main}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Params & atom\_acc & reaction\_acc \\",
        r"\midrule",
    ]
    atom_means = {k: pm["atom_acc"]["mean"] for k, pm in summary["per_model"].items()}
    valid = [v for v in atom_means.values() if not math.isnan(v)]
    best_a = max(valid) if valid else float("nan")
    for mkey, pm in summary["per_model"].items():
        atom_s = (
            f"{pm['atom_acc']['mean']:.4f}\\,$\\pm$\\,{pm['atom_acc']['ci95']:.4f}"
            if not math.isnan(pm["atom_acc"]["mean"])
            else "n/a"
        )
        rxn_s = (
            f"{pm['reaction_acc']['mean']:.4f}\\,$\\pm$\\,{pm['reaction_acc']['ci95']:.4f}"
            if not math.isnan(pm["reaction_acc"]["mean"])
            else "n/a"
        )
        if (
            not math.isnan(pm["atom_acc"]["mean"])
            and abs(pm["atom_acc"]["mean"] - best_a) < 1e-9
        ):
            atom_s = r"\textbf{" + atom_s + "}"
        lines.append(
            _latex_escape(MODEL_KEYS[mkey])
            + " & "
            + f"{pm['params']:,}"
            + " & "
            + atom_s
            + " & "
            + rxn_s
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_error_distribution_md(path: Path, error_summary: Dict, args):
    """Per-architecture error-distribution report."""
    out = ["# Error-distribution analysis (test set)", ""]
    out += [
        "Pooled across all seeds per architecture, on the held-out test",
        "set. The `concentration ratio` is observed reaction_acc divided",
        "by the i.i.d. counterfactual: what reaction_acc would be if",
        "atom errors were independent within reactions, given each",
        "architecture's per-atom error rate.",
        "",
        "  ratio = 1 → errors scattered as predicted by independence",
        "  ratio > 1 → errors **concentrated** in fewer reactions",
        "              (good for reaction_acc relative to atom_acc)",
        "  ratio < 1 → errors **scattered** more than independent",
        "              (bad for reaction_acc relative to atom_acc)",
        "",
        "## Headline stats",
        "",
        "| model | atom_acc | reaction_acc obs | iid pred | ratio | mean errs / failed rxn |",
        "|---|---|---|---|---:|---:|",
    ]
    for mkey, s in error_summary.items():
        out.append(
            f"| {MODEL_KEYS[mkey]} "
            f"| {s['atom_acc']:.4f} "
            f"| {s['reaction_acc_observed']:.4f} "
            f"| {s['reaction_acc_iid']:.4f} "
            f"| {s['concentration_ratio']:.2f} "
            f"| {s['mean_errors_per_failed_rxn']:.2f} |"
        )
    out += [
        "",
        "## Error-count distribution (fraction of reactions per bucket)",
        "",
        "| model | "
        + " | ".join(f"{lab} errors" for lab in ERROR_BUCKET_LABELS)
        + " |",
        "|---" + "|---:" * len(ERROR_BUCKET_LABELS) + "|",
    ]
    for mkey, s in error_summary.items():
        cells = " | ".join(f"{f:.3f}" for f in s["bucket_fracs"])
        out.append(f"| {MODEL_KEYS[mkey]} | {cells} |")
    out += [
        "",
        "## How to read this",
        "",
        "If GSC architectures have a **higher concentration ratio** than",
        "the GCN+aggregator pipelines, that is direct evidence for the",
        "mechanism hypothesis: GSC's interleaved attention couples atoms",
        "within a reaction, so when it errs it errs on *several atoms in",
        "one reaction* rather than scattering single-atom errors across",
        "many reactions. Pipelines, with one broadcast vector per",
        "molecule and no atom-level cross-coupling, make per-atom",
        "decisions more independently — which scatters errors and",
        "depresses reaction_acc disproportionately.",
        "",
        "Compare the bucket distributions: a more concentrated",
        "architecture has higher mass at the 0-error bucket (more",
        "fully-correct reactions) AND at the 4-7+ error buckets (when it",
        "fails it fails harder), with less mass at the 1-2 error buckets.",
        "",
    ]
    path.write_text("\n".join(out) + "\n")


def write_error_distribution_csv(path: Path, error_summary: Dict):
    """Flat CSV with one row per architecture for re-analysis."""
    rows: List[Dict] = []
    for mkey, s in error_summary.items():
        row = {
            "model_key": mkey,
            "model_name": MODEL_KEYS[mkey],
            "n_rxns": s["n_rxns"],
            "n_atoms_total": s["n_atoms_total"],
            "atom_acc": s["atom_acc"],
            "reaction_acc_observed": s["reaction_acc_observed"],
            "reaction_acc_iid": s["reaction_acc_iid"],
            "concentration_ratio": s["concentration_ratio"],
            "mean_errors_per_failed_rxn": s["mean_errors_per_failed_rxn"],
        }
        for lab, c, f in zip(
            ERROR_BUCKET_LABELS, s["bucket_counts"], s["bucket_fracs"]
        ):
            row[f"bucket_{lab}_count"] = c
            row[f"bucket_{lab}_frac"] = f
        rows.append(row)
    write_csv(path, rows)


def write_error_distribution_latex(path: Path, error_summary: Dict, args):
    """Per-architecture error-distribution LaTeX table."""
    lines = [
        r"% Auto-generated by atom_mapping_benchmark.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Error distribution on USPTO-50K test set, "
            r"pooled across "
            + str(args.seeds)
            + r" seeds per architecture. "
            + r"\emph{Obs}: empirical reaction\_acc. "
            + r"\emph{IID}: counterfactual reaction\_acc if atom "
            + r"errors were independent within reactions at the "
            + r"observed per-atom error rate. "
            + r"\emph{Ratio}: Obs $/$ IID — values $>1$ mean errors "
            + r"are concentrated in fewer reactions than independence "
            + r"would predict.}"
        ),
        r"\label{tab:atom_mapping_errors}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & atom\_acc & rxn\_acc obs & rxn\_acc iid & "
        r"ratio & mean errs / fail \\",
        r"\midrule",
    ]
    for mkey, s in error_summary.items():
        lines.append(
            _latex_escape(MODEL_KEYS[mkey])
            + " & "
            + f"{s['atom_acc']:.4f}"
            + " & "
            + f"{s['reaction_acc_observed']:.4f}"
            + " & "
            + f"{s['reaction_acc_iid']:.4f}"
            + " & "
            + f"{s['concentration_ratio']:.2f}"
            + " & "
            + f"{s['mean_errors_per_failed_rxn']:.2f}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


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
        "--data-dir",
        default="data/uspto50k",
        help="Directory containing the predefined USPTO-50K split CSVs "
        "(default file names: raw_train.csv, raw_val.csv, raw_test.csv; "
        "override with --train-file etc.).",
    )
    p.add_argument("--train-file", default="raw_train.csv")
    p.add_argument("--val-file", default="raw_val.csv")
    p.add_argument("--test-file", default="raw_test.csv")
    p.add_argument(
        "--rxn-col",
        default=None,
        help="CSV column with the atom-mapped rxn SMILES "
        "(default: auto-detect; standard GLN name is "
        "'reactants>reagents>production').",
    )
    p.add_argument("--id-col", default="id")
    p.add_argument("--out-dir", default="results/uspto50k_run")
    p.add_argument("--quick", action="store_true")

    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_KEYS.keys()),
        choices=list(MODEL_KEYS.keys()),
    )
    p.add_argument(
        "--framing",
        choices=list(FRAMING_KEYS),
        default="siamese",
        help="Set keying for the encoder. siamese (default): "
        "mol_to_set = rxn*2 + side, encoder is within-side; the "
        "assignment head does ALL cross-side work. joint: "
        "mol_to_set = rxn, encoder sees both sides as one set per "
        "reaction. Empirically (Golden, USPTO-50K) the framing makes "
        "no detectable difference; siamese is the cleaner baseline.",
    )

    # Subsampling (USPTO-50K is much larger than Golden; useful for
    # quick architectural experiments before committing to a full run).
    p.add_argument(
        "--max-train-size",
        type=int,
        default=None,
        help="Cap the training set to first K reactions (None = all).",
    )
    p.add_argument(
        "--max-val-size",
        type=int,
        default=None,
        help="Cap the validation set to first K reactions.",
    )
    p.add_argument(
        "--max-test-size",
        type=int,
        default=None,
        help="Cap the test set to first K reactions.",
    )

    p.add_argument(
        "--max-reactant-atoms",
        type=int,
        default=None,
        help="Drop reactions with > K reactant atoms (memory/speed control).",
    )
    p.add_argument(
        "--max-product-atoms",
        type=int,
        default=None,
        help="Drop reactions with > K product atoms.",
    )
    p.add_argument(
        "--drop-unmapped-reactant-components",
        action="store_true",
        help="Drop reactant components (e.g. spectator reagents) with "
        "zero mapped atoms. Default False = keep them as set context.",
    )
    p.add_argument(
        "--canonicalize",
        action="store_true",
        help="RetroXpert-style canonical reorder of atoms (strip map "
        "numbers, get canonical rank, reorder, restore maps). For "
        "GNN-based atom mapping with mapping numbers as targets only "
        "this is not strictly needed — atom order does not enter "
        "computation through any path the model can use — but the "
        "flag is provided for cross-paper comparability.",
    )

    # Architecture
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--equalize-params",
        choices=["none", "smaller", "larger"],
        default="none",
    )
    p.add_argument(
        "--no-side-identity",
        dest="use_side_identity",
        action="store_false",
        default=True,
        help="Disable the atom-level side embedding (ablation).",
    )

    # Training (defaults tuned for ~40K train; was 200/30 for 1.5K
    # Golden, would be wasteful here).
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)

    # Progress
    p.add_argument("--log-every-epoch", type=int, default=1)

    args = p.parse_args()
    if args.quick:
        args.epochs = 10
        args.patience = 3
        args.seeds = 1
        args.batch_size = 4
        args.hidden = 64
        args.layers = 2
        if args.max_train_size is None:
            args.max_train_size = 1000
        if args.max_val_size is None:
            args.max_val_size = 200
        if args.max_test_size is None:
            args.max_test_size = 200
    return args


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"=== atom_mapping_benchmark.py ===")
    print(f"  device       = {device}")
    print(f"  out-dir      = {out_dir}")
    print(f"  data-dir     = {args.data_dir}")
    print(f"  models       = {args.models}")
    print(f"  framing      = {args.framing}")

    # Load predefined train/val/test splits.
    train, val, test = load_uspto50k(
        args.data_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        rxn_col=args.rxn_col,
        id_col=args.id_col,
        drop_unmapped_reactant_components=args.drop_unmapped_reactant_components,
        canonicalize=args.canonicalize,
        max_train=args.max_train_size,
        max_val=args.max_val_size,
        max_test=args.max_test_size,
        max_reactant_atoms=args.max_reactant_atoms,
        max_product_atoms=args.max_product_atoms,
    )
    print(f"\n  train/val/test = {len(train)}/{len(val)}/{len(test)}")
    if not train or not test:
        raise RuntimeError("Empty train or test split.")

    # Per-model hidden dim under --equalize-params.
    feat_dim = ATOM_FEAT_DIM
    hiddens_by_model, param_counts = equalize_models(args.models, args, feat_dim)

    print(f"\n[equalize-params={args.equalize_params}] per-model budgets:")
    for mkey, n in param_counts.items():
        h = hiddens_by_model[mkey]
        print(f"  {MODEL_KEYS[mkey]:<28}  hidden={h:>4}  params={n:>9,}")

    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "models": {k: MODEL_KEYS[k] for k in args.models},
            "param_counts": param_counts,
            "hiddens_by_model": hiddens_by_model,
            "split_sizes": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "atom_feat_dim": ATOM_FEAT_DIM,
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

    # ---- Training loop ----
    raw_scores: Dict[str, Dict[str, List[float]]] = {}
    timings: Dict[str, List[float]] = {}
    flat_rows: List[Dict] = []
    history_all: Dict[str, Dict[int, List]] = {}
    # Per-model: pooled (n_atoms, n_errors) tuples across all seeds.
    per_rxn_pooled: Dict[str, List[Tuple[int, int]]] = {}

    for mkey in args.models:
        print(f"\n--- {MODEL_KEYS[mkey]} ---")
        raw_scores[mkey] = {"atom_acc": [], "reaction_acc": []}
        timings[mkey] = []
        history_all[mkey] = {}
        per_rxn_pooled[mkey] = []

        for sidx in range(args.seeds):
            seed = args.seed_offset + sidx
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = build_model(
                mkey,
                args.framing,
                args.layers,
                hiddens_by_model[mkey],
                args.num_heads,
                args.dropout,
                atom_feat_dim=feat_dim,
                use_side_identity=args.use_side_identity,
            ).to(device)
            n_params = count_params(model)
            progress_prefix = f"[{MODEL_KEYS[mkey]} seed={seed}] "
            metrics, elapsed, hist = train_one(
                model,
                train,
                val,
                test,
                args,
                seed,
                device,
                progress_prefix=progress_prefix,
            )
            print(
                f"  seed={seed:>2}  test atom_acc={metrics['atom_acc']:.4f}  "
                f"reaction_acc={metrics['reaction_acc']:.4f}  "
                f"({elapsed:.1f}s, {n_params:,} params)"
            )
            raw_scores[mkey]["atom_acc"].append(metrics["atom_acc"])
            raw_scores[mkey]["reaction_acc"].append(metrics["reaction_acc"])
            timings[mkey].append(elapsed)
            history_all[mkey][seed] = hist
            # Pool per-rxn errors across seeds. Note: pooling treats
            # each seed's eval as independent samples on the same test
            # set; this enlarges the bucket histogram (and stabilizes
            # the concentration_ratio) without conflating seeds.
            per_rxn_pooled[mkey].extend(metrics.get("per_rxn", []))
            flat_rows.append(
                {
                    "model_key": mkey,
                    "model_name": MODEL_KEYS[mkey],
                    "framing": args.framing,
                    "seed": seed,
                    "atom_acc": metrics["atom_acc"],
                    "reaction_acc": metrics["reaction_acc"],
                    "params": n_params,
                    "wall_seconds": round(elapsed, 2),
                }
            )
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- Aggregate ----
    per_model: Dict[str, Dict] = {}
    for mkey, s in raw_scores.items():
        if not s["atom_acc"]:
            continue
        agg: Dict = {}
        for metric in ("atom_acc", "reaction_acc"):
            mean, half = t_ci_95(s[metric])
            agg[metric] = {"mean": mean, "ci95": half, "scores": s[metric]}
        agg["params"] = param_counts[mkey]
        agg["wall_seconds_mean"] = float(np.mean(timings[mkey]))
        per_model[mkey] = agg

    # ---- Error-distribution analysis (pooled across seeds per model) ----
    error_summary: Dict[str, Dict] = {}
    for mkey in args.models:
        if mkey in per_model:
            error_summary[mkey] = analyze_error_distribution(per_rxn_pooled[mkey])

    summary = {"per_model": per_model, "error_summary": error_summary}

    write_json(
        out_dir / "raw.json",
        {
            "summary": summary,
            "raw_scores": raw_scores,
            "timings": timings,
            "history": history_all,
            "per_rxn_pooled": per_rxn_pooled,
        },
    )
    write_csv(out_dir / "summary.csv", flat_rows)
    write_markdown(out_dir / "summary.md", summary, args)
    write_latex(out_dir / "summary.tex", summary, args)
    write_error_distribution_md(out_dir / "error_distribution.md", error_summary, args)
    write_error_distribution_csv(out_dir / "error_distribution.csv", error_summary)
    write_error_distribution_latex(
        out_dir / "error_distribution.tex", error_summary, args
    )

    # Console summary
    print(f"\n  ===== test summary =====")
    atom_means = {k: pm["atom_acc"]["mean"] for k, pm in per_model.items()}
    valid = [v for v in atom_means.values() if not math.isnan(v)]
    best_a = max(valid) if valid else float("nan")
    for mkey, pm in per_model.items():
        marker = " *" if abs(pm["atom_acc"]["mean"] - best_a) < 1e-9 else "  "
        print(
            f"   {marker} {MODEL_KEYS[mkey]:<28} "
            f"atom={pm['atom_acc']['mean']:.4f}±{pm['atom_acc']['ci95']:.4f}  "
            f"rxn={pm['reaction_acc']['mean']:.4f}±{pm['reaction_acc']['ci95']:.4f}  "
            f"({pm['params']:,} params)"
        )
    if error_summary:
        print(f"\n  ===== error distribution (test, pooled across seeds) =====")
        for mkey, s in error_summary.items():
            print(
                f"   {MODEL_KEYS[mkey]:<28} "
                f"obs={s['reaction_acc_observed']:.4f}  "
                f"iid={s['reaction_acc_iid']:.4f}  "
                f"ratio={s['concentration_ratio']:.2f}  "
                f"errs/fail={s['mean_errors_per_failed_rxn']:.2f}"
            )
    print(f"\nAll outputs in: {out_dir}")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
