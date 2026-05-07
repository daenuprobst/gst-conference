"""
graph_set_conv_painn.py
=======================

PaiNN-backed GraphSetConv blocks for molecular conformer-set tasks.

Self-contained module. No imports from your existing GCN-based blocks; the
PaiNN core is reproduced here so this file can be dropped into a fresh
project.

Architecture variants (all use the same PaiNN per-graph backbone):
    GraphSetConvPaiNNBroadcast  — uniform per-graph set context to all atoms.
    GraphSetConvPaiNNCrossAttn  — per-node cross-attention to the post-set-
                                   attention token of the atom's home graph.

These mirror the GCN-version's `node_set_mode` flag, exposed here as two
separate classes for clarity (no flag).

Equivariance contract
---------------------
PaiNN node features split into:
    s_i  in R^F       scalar (rotation-invariant)
    v_i  in R^{F,3}   vector (rotation-equivariant)

Set context is computed from scalar features only (pooled over atoms within
each graph) and routed back to scalar features only. Vector features are
NEVER touched by set-level operations. This preserves end-to-end equivariance.

If you want to ablate the equivariance constraint (set context routes into
vectors as well, breaking equivariance), set node_set_routing="both" — the
flag exists as an ablation lever, but use scalar-only by default.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

# torch_scatter is the recommended C++/CUDA backend; if unavailable we fall
# back to pure-torch equivalents. The fallbacks are slower but functionally
# identical and let the script run on environments where torch_scatter is
# not installed (e.g. some CPU-only sandboxes).
try:
    from torch_scatter import scatter
except ImportError:

    def scatter(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int = None,
        reduce: str = "add",
    ) -> torch.Tensor:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        if reduce == "add" or reduce == "sum":
            out = src.new_zeros(out_shape)
            shape = [1] * src.dim()
            shape[dim] = -1
            idx = index.view(shape).expand_as(src)
            out.scatter_add_(dim, idx, src)
            return out
        if reduce == "mean":
            out = src.new_zeros(out_shape)
            shape = [1] * src.dim()
            shape[dim] = -1
            idx = index.view(shape).expand_as(src)
            out.scatter_add_(dim, idx, src)
            count = src.new_zeros(dim_size)
            count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
            count_shape = [1] * src.dim()
            count_shape[dim] = dim_size
            out = out / count.view(count_shape).clamp_min(1.0)
            return out
        raise NotImplementedError(f"scatter reduce={reduce!r}")


# =============================================================================
# Helpers (radial basis, cosine cutoff, drop-path, activations)
# =============================================================================


class GaussianRBF(nn.Module):
    """Distance -> radial basis. Standard PaiNN/SchNet construction."""

    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        offsets = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("offsets", offsets)
        self.coeff = -0.5 / ((cutoff / num_rbf) ** 2)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        diff = r.unsqueeze(-1) - self.offsets
        return torch.exp(self.coeff * diff.pow(2))


def cosine_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Smooth cutoff envelope. Zero outside `cutoff`, 1 at r=0."""
    return torch.where(
        r < cutoff,
        0.5 * (torch.cos(math.pi * r / cutoff) + 1.0),
        torch.zeros_like(r),
    )


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


def _act(name: str = "silu") -> nn.Module:
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
    }.get(name, nn.SiLU)()


# =============================================================================
# PaiNN per-graph core (one block)
# =============================================================================


class PaiNNCore(nn.Module):
    """One PaiNN block: equivariant scalar+vector message passing.

    Reference:  Schuett, Unke, Gastegger, "Equivariant message passing for
    the prediction of tensorial properties and molecular spectra", ICML 2021.

    Conventions:
      s : [N, F]      scalar features
      v : [N, F, 3]   vector features (rotation-equivariant)
      pos : [N, 3]    atom positions
      edge_index : [2, E]   directed; we treat src->dst messages, do not
                            symmetrize internally.
    """

    def __init__(self, dim: int, num_rbf: int, cutoff: float, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.cutoff = cutoff
        self.rbf = GaussianRBF(num_rbf, cutoff)

        # Edge filter: RBF -> 3F (three gating channels)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, dim),
            nn.SiLU(),
            nn.Linear(dim, 3 * dim),
        )
        # Source-node scalar transform -> 3F (three message channels)
        self.phi = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 3 * dim),
        )
        # Update step: gated equivariant.
        self.U = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)
        self.update_net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 3 * dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, s, v, pos, edge_index):
        src, dst = edge_index
        rij = pos[dst] - pos[src]  # [E, 3]
        r = rij.norm(dim=-1).clamp_min(1e-12)  # [E]
        rij_hat = rij / r.unsqueeze(-1)  # unit vec
        env = self.rbf(r) * cosine_cutoff(r, self.cutoff).unsqueeze(-1)
        f = self.filter_net(env)  # [E, 3F]
        m = self.phi(s[src]) * f  # [E, 3F]
        m_s, m_vv, m_vd = m.split(self.dim, dim=-1)  # each [E, F]

        # Scalar message
        ds = scatter(m_s, dst, dim=0, dim_size=s.size(0), reduce="add")
        # Vector message: m_vv * v_src + m_vd * rij_hat (lifted to vector)
        v_src = v[src]  # [E, F, 3]
        msg_v = m_vv.unsqueeze(-1) * v_src + m_vd.unsqueeze(-1) * rij_hat.unsqueeze(
            1
        )  # [E, F, 3]
        dv = scatter(msg_v, dst, dim=0, dim_size=v.size(0), reduce="add")

        s = s + self.dropout(ds)
        v = v + dv

        # Update step (gated equivariant block)
        Uv = self.U(v.transpose(-1, -2)).transpose(-1, -2)  # [N, F, 3]
        Vv = self.V(v.transpose(-1, -2)).transpose(-1, -2)  # [N, F, 3]
        Vv_norm = Vv.norm(dim=-1)  # [N, F]
        upd_input = torch.cat([s, Vv_norm], dim=-1)  # [N, 2F]
        a = self.update_net(upd_input)  # [N, 3F]
        a_vv, a_sv, a_ss = a.split(self.dim, dim=-1)
        # vector update: a_vv * Uv  (gated equivariant)
        v = v + a_vv.unsqueeze(-1) * Uv
        # scalar update: a_ss + a_sv * <Uv, Vv>
        UV_dot = (Uv * Vv).sum(dim=-1)  # [N, F]
        s = s + a_ss + a_sv * UV_dot
        return s, v


# =============================================================================
# Per-node cross-attention helper
# =============================================================================


class PerNodeMultiTokenCrossAttention(nn.Module):
    """Per-atom cross-attention to the K post-set-attention tokens of the
    node's own graph. Generalizes the K=1 case used in CrossAttn variant."""

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, qk_norm: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x_nodes, graph_tokens, node_graph_id):
        N = x_nodes.shape[0]
        G, K, D = graph_tokens.shape
        H, Hd = self.num_heads, self.head_dim
        q = self.q_proj(x_nodes).view(N, H, 1, Hd)
        kv = self.kv_proj(graph_tokens)
        k_g, v_g = kv.chunk(2, dim=-1)
        k_g = k_g.view(G, K, H, Hd).transpose(1, 2)
        v_g = v_g.view(G, K, H, Hd).transpose(1, 2)
        k = k_g[node_graph_id]
        v = v_g[node_graph_id]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).reshape(N, D))


# =============================================================================
# Set-level pooling helper (atoms -> per-conformer scalar embedding)
# =============================================================================


def conformer_pool(
    s: torch.Tensor, batch: torch.Tensor, reduce: str = "add"
) -> torch.Tensor:
    """Pool atom-level scalars to per-conformer (per-graph) scalars.
    Vectors are not pooled — they cannot be aggregated across atoms while
    preserving equivariance unless a frame is chosen."""
    return scatter(s, batch, dim=0, reduce=reduce)


# =============================================================================
# Broadcast variant: uniform per-graph set context (PaiNN backbone)
# =============================================================================


class GraphSetConvPaiNNBroadcast(nn.Module):
    """Broadcast variant: PaiNN per-graph propagation, single-token
    attentional pool over atoms, set-level self-attention over per-graph
    tokens, broadcast back to all atoms of the same graph (uniform per
    graph), gated residual.

    All set-level computations operate on SCALAR features only.
    """

    def __init__(
        self,
        dim: int,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        num_heads: int = 4,
        ffn_multiplier: float = 2.0,
        mhsa_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        use_gating: bool = True,
        drop_path: float = 0.0,
        node_set_routing: str = "scalar",
    ):
        super().__init__()
        assert node_set_routing in ("scalar", "both"), node_set_routing
        self.node_set_routing = node_set_routing
        self.use_gating = use_gating
        if num_heads == 0:
            num_heads = max(1, dim // 16)
        assert dim % num_heads == 0

        self.painn = PaiNNCore(dim, num_rbf, cutoff, dropout=ffn_dropout)

        # Single-token attentional pool over atoms (gate-attention).
        self.gate_pool = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)
        )
        self.value_proj = nn.Identity()

        # Set-level transformer: pre-norm, MHSA, FFN.
        self.ln1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            dim, num_heads, dropout=mhsa_dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_multiplier)),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(int(dim * ffn_multiplier), dim),
            nn.Dropout(ffn_dropout),
        )
        self.drop_path_attn = DropPath(drop_path)
        self.drop_path_ffn = DropPath(drop_path)

        # NOTE: deliberately no `ln_post` here. The classic post-norm at the
        # end of the set-attn block re-normalized the per-graph tokens before
        # they got broadcast back to atoms, which strips the inter-graph
        # statistics (means, magnitudes) that downstream layers might need.
        # Pre-norm inside the MHA/FFN blocks already provides the
        # stabilization we need; the residual stream stays un-normalized so
        # tasks whose labels depend on inter-graph statistics (e.g. bilinear
        # over-squashing) still have the signal to read off.

        # LayerNorm on the per-graph context before residual addition.
        # Without this, the residual stream's scalar magnitude doubles each
        # layer (graph_info[batch] ~ |s| because attn-pool is mean-like, so
        # s + graph_info ~ 2*s). At depth >=4 this compounds catastrophically.
        # This is a separate concern from `ln_post`: it controls the per-node
        # residual stream's growth, not the per-graph token stream.
        self.ln_graph_info = nn.LayerNorm(dim)

        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            nn.init.constant_(self.gate[-1].bias, -1.0)

        # Optional gate for vector channel (only used if node_set_routing="both").
        if node_set_routing == "both":
            self.vec_gate = nn.Linear(dim, dim, bias=True)
            nn.init.constant_(self.vec_gate.bias, -1.0)

    def _attn_pool(self, s, batch):
        gate_logits = self.gate_pool(s)  # [N, 1]
        s_dense, mask = to_dense_batch(s, batch)  # [G, Nmax, D]
        g_dense, _ = to_dense_batch(gate_logits, batch)  # [G, Nmax, 1]
        g_dense = g_dense.masked_fill(~mask.unsqueeze(-1), -1e9)
        w = F.softmax(g_dense, dim=1)  # [G, Nmax, 1]
        return (w * s_dense).sum(dim=1)  # [G, D]

    def forward(self, s, v, pos, edge_index, batch, set_batch):
        # Per-graph propagation.
        s, v = self.painn(s, v, pos, edge_index)

        # Per-graph scalar embedding via attention pool.
        z_graph = self._attn_pool(s, batch)  # [G, D]

        # Set-level self-attention over per-graph tokens.
        z_dense, mask = to_dense_batch(z_graph, set_batch)
        zn = self.ln1(z_dense)
        z_attn, _ = self.mha(zn, zn, zn, key_padding_mask=~mask, need_weights=False)
        z_dense = z_dense + self.drop_path_attn(z_attn)
        z_dense = z_dense + self.drop_path_ffn(self.ffn(self.ln2(z_dense)))
        z_set = z_dense[mask]  # [G, D]

        # Broadcast set context back to atoms (uniform per graph).
        graph_info = self.ln_graph_info(z_graph[batch])  # [N, D]
        set_info = z_set[batch]  # [N, D]

        if self.use_gating:
            g_set = torch.sigmoid(self.gate(torch.cat([s, set_info], dim=-1)))
            s = s + graph_info + g_set * set_info
        else:
            s = s + graph_info + set_info

        # Optional equivariance-breaking ablation: route set context to vectors
        # via a *learned scalar gate*. This still rotates correctly
        # multiplicatively (the magnitude is invariant), but the magnitude
        # becomes set-dependent, which is what we want to demonstrate breaks
        # the equivariance contract when chained with downstream layers.
        if self.node_set_routing == "both":
            vg = torch.sigmoid(self.vec_gate(set_info))  # [N, D]
            v = v * (1.0 + vg).unsqueeze(-1)
        return s, v


# =============================================================================
# CrossAttn variant: per-node cross-attention to single-token-per-graph (PaiNN backbone)
# =============================================================================


class GraphSetConvPaiNNCrossAttn(nn.Module):
    """Cross-attention variant: PaiNN propagation; per-graph attention
    pool produces ONE token per graph; set-level self-attention; per-node
    cross-attention to the post-set-attention token of the atom's home
    graph (K=1 case).
    """

    def __init__(
        self,
        dim: int,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        num_heads: int = 4,
        ffn_multiplier: float = 2.0,
        mhsa_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        use_gating: bool = True,
        drop_path: float = 0.0,
        node_set_routing: str = "scalar",
    ):
        super().__init__()
        assert node_set_routing in ("scalar", "both"), node_set_routing
        self.node_set_routing = node_set_routing
        self.use_gating = use_gating
        if num_heads == 0:
            num_heads = max(1, dim // 16)
        assert dim % num_heads == 0

        self.painn = PaiNNCore(dim, num_rbf, cutoff, dropout=ffn_dropout)

        self.gate_pool = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)
        )

        self.ln1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            dim, num_heads, dropout=mhsa_dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_multiplier)),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(int(dim * ffn_multiplier), dim),
            nn.Dropout(ffn_dropout),
        )
        # NOTE: deliberately no `ln_post` here — see the note in the
        # Broadcast class.
        self.drop_path_attn = DropPath(drop_path)
        self.drop_path_ffn = DropPath(drop_path)

        self.ln_q = nn.LayerNorm(dim)
        # K=1 case of multi-token CA: per-atom cross-attention to the
        # single post-set-attn token of the atom's home graph.
        self.node_ca = PerNodeMultiTokenCrossAttention(
            dim,
            num_heads,
            dropout=mhsa_dropout,
            qk_norm=False,
        )

        # LayerNorm on graph_info before residual (see Broadcast class).
        self.ln_graph_info = nn.LayerNorm(dim)

        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            nn.init.constant_(self.gate[-1].bias, -1.0)

        if node_set_routing == "both":
            self.vec_gate = nn.Linear(dim, dim, bias=True)
            nn.init.constant_(self.vec_gate.bias, -1.0)

    def _attn_pool(self, s, batch):
        gate_logits = self.gate_pool(s)
        s_dense, mask = to_dense_batch(s, batch)
        g_dense, _ = to_dense_batch(gate_logits, batch)
        g_dense = g_dense.masked_fill(~mask.unsqueeze(-1), -1e9)
        w = F.softmax(g_dense, dim=1)
        return (w * s_dense).sum(dim=1)

    def forward(self, s, v, pos, edge_index, batch, set_batch):
        s, v = self.painn(s, v, pos, edge_index)
        z_graph = self._attn_pool(s, batch)  # [G, D]

        # Set-level self-attention over per-graph tokens.
        z_dense, mask = to_dense_batch(z_graph, set_batch)
        zn = self.ln1(z_dense)
        z_attn, _ = self.mha(zn, zn, zn, key_padding_mask=~mask, need_weights=False)
        z_dense = z_dense + self.drop_path_attn(z_attn)
        z_dense = z_dense + self.drop_path_ffn(self.ffn(self.ln2(z_dense)))
        z_set_per_graph = z_dense[mask]  # [G, D]

        graph_info = self.ln_graph_info(z_graph[batch])  # [N, D]

        # Per-atom cross-attention to its home graph's single set-token.
        # Reshape per-graph tokens into [G, 1, D] so we can use the
        # multi-token CA module uniformly.
        graph_tokens = z_set_per_graph.unsqueeze(1)  # [G, 1, D]
        set_info = self.node_ca(self.ln_q(s), graph_tokens, batch)

        if self.use_gating:
            g_set = torch.sigmoid(self.gate(torch.cat([s, set_info], dim=-1)))
            s = s + graph_info + g_set * set_info
        else:
            s = s + graph_info + set_info

        if self.node_set_routing == "both":
            vg = torch.sigmoid(self.vec_gate(set_info))
            v = v * (1.0 + vg).unsqueeze(-1)
        return s, v


# =============================================================================
# Pure-pipeline baselines (for comparison): PaiNN encoder, then set head.
# =============================================================================


class PaiNNEncoder(nn.Module):
    """Stack of L PaiNN blocks. Returns final scalar features per atom.
    No set-level operations; this is the baseline encoder for the
    PaiNN+DeepSets and PaiNN+SetTransformer pipelines."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_rbf: int,
        cutoff: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cores = nn.ModuleList(
            [
                PaiNNCore(dim, num_rbf, cutoff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, s, v, pos, edge_index):
        for core in self.cores:
            s, v = core(s, v, pos, edge_index)
        return s, v


class DeepSetsHead(nn.Module):
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

    def forward(self, z_graph, set_batch, log_boltz_w=None):
        # log_boltz_w accepted for API uniformity; DeepSets ignores it
        # (sum-pool is its defining operation).
        h = self.phi(z_graph)
        z_set = scatter(h, set_batch, dim=0, reduce="add")
        return self.rho(z_set)


class SetTransformerHead(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.1, ffn_mult: int = 2
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

    def forward(self, z_graph, set_batch, log_boltz_w=None):
        # log_boltz_w accepted for API uniformity; SetTransformer ignores it.
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


class BoltzmannPool(nn.Module):
    """Physics-grounded baseline: graph -> set aggregator that uses the
    Boltzmann weights directly. Each per-graph embedding is weighted by
    softmax(log_boltz_w) within its set, then summed.

    No learnable parameters in the pooling itself; an optional MLP
    transforms the resulting set embedding for fair comparison with
    DeepSets/SetTransformer (which both have phi/rho-like learnable bits).
    Set this MLP to identity by passing `head_mlp=False`.

    This is the closest the model architecture gets to "literal Boltzmann
    averaging" — if KRAKEN's targets really decompose as
        y_target = sum_k softmax(log_w_k) * f(conformer_k)
    then this baseline recovers exactly that with `f` learned by the
    encoder. Strong baseline for any architecture that claims to do better.
    """

    def __init__(self, dim: int, dropout: float = 0.1, head_mlp: bool = True):
        super().__init__()
        if head_mlp:
            self.proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, z_graph, set_batch, log_boltz_w):
        if log_boltz_w is None:
            raise RuntimeError(
                "BoltzmannPool requires log_boltz_w; got None. The collate "
                "should pack per-conformer log-Boltzmann-weights into the "
                "PackedBatch."
            )
        # Compute per-set softmax(log_boltz_w). Use scatter-style
        # softmax: subtract per-set max for stability, exp, normalize.
        z_dense, mask = to_dense_batch(z_graph, set_batch)  # [S, K, D]
        lw_dense, _ = to_dense_batch(log_boltz_w.unsqueeze(-1), set_batch)  # [S, K, 1]
        lw_dense = lw_dense.squeeze(-1)
        # Mask out padded positions before softmax.
        lw_masked = lw_dense.masked_fill(~mask, float("-inf"))
        w = torch.softmax(lw_masked, dim=1).unsqueeze(-1)  # [S, K, 1]
        z_set = (w * z_dense).sum(dim=1)  # [S, D]
        return self.proj(z_set)


class AttentionalSetPool(nn.Module):
    """Learned attention pool over conformers: graph -> set. Mirrors the
    AttentionalAggregation pattern from torch_geometric. Each per-graph
    embedding is given a learned scalar gate, gates are softmaxed within
    each set, and the weighted sum is the set embedding.

    Unlike BoltzmannPool, the weights are LEARNED rather than fixed by
    physics — but the form is identical, which means at convergence this
    pool can in principle reproduce Boltzmann weighting if that's optimal.
    Used for GSC's graph -> set step so that the comparison with pipeline
    baselines (DeepSets, SetTransformer) is on the strength of the
    architectures' inductive biases, not on which one happens to have a
    learnable aggregator.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, z_graph, set_batch, log_boltz_w=None):
        # log_boltz_w accepted for API uniformity; not used by this pool.
        # (If you want attention-over-conformers conditioned on energy,
        # concat log_boltz_w into z_graph BEFORE this pool — that's done
        # by the energy-feature step in the model.)
        gate_logits = self.gate(z_graph)  # [G, 1]
        gates_dense, mask = to_dense_batch(gate_logits, set_batch)  # [S, K, 1]
        z_dense, _ = to_dense_batch(z_graph, set_batch)  # [S, K, D]
        gates_dense = gates_dense.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(gates_dense, dim=1)  # [S, K, 1]
        z_set = (w * z_dense).sum(dim=1)  # [S, D]
        return self.proj(z_set)


class EnergyFeature(nn.Module):
    """Tiny module that lifts a scalar log-Boltzmann-weight to a `dim`-vector
    suitable for adding to z_graph as a per-conformer feature. Uses a small
    Fourier expansion + linear so the model can learn smooth functions of
    log_w without numerical issues at large negative log-weights.
    """

    def __init__(self, dim: int, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        # Fourier features: log_w * pi * 2^[0..num_freqs-1]
        freqs = torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", torch.pi * (2.0**freqs), persistent=False)
        self.proj = nn.Linear(2 * num_freqs + 1, dim)

    def forward(self, log_boltz_w: torch.Tensor) -> torch.Tensor:
        # log_boltz_w: [G]; output: [G, dim]
        x = log_boltz_w.unsqueeze(-1)  # [G, 1]
        sinusoids = torch.cat(
            [torch.sin(x * self.freqs), torch.cos(x * self.freqs)],
            dim=-1,
        )  # [G, 2*F]
        feats = torch.cat([x, sinusoids], dim=-1)  # [G, 1+2*F]
        return self.proj(feats)
