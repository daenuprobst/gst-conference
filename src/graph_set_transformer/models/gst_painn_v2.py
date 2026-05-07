"""
graph_set_conv_painn.py
=======================

PaiNN-backed GraphSetConv blocks for molecular conformer-set tasks (redesigned).

Self-contained module. No imports from the GCN-based blocks; the PaiNN core is
reproduced here so this file can be dropped into a fresh project.

Architecture variants (same PaiNN per-graph backbone, both updated):
    GraphSetConvPaiNNBroadcast  - uniform per-graph set context to all atoms.
    GraphSetConvPaiNNCrossAttn  - per-atom cross-attention to the
                                   post-set-attention token of the atom's home graph.

Modernizations vs. the previous version
---------------------------------------
- LayerScale (CaiT) + DropPath consistently on every set-level residual branch.
- Optional per-atom scalar FFN after PaiNN (GraphGPS-style MPNN -> FFN, scalar-
  only; vectors stay equivariant).
- Per-graph pool can be PMA (Set Transformer) instead of single-scalar-gate.
- Pre-LN on KV in cross-attention (was asymmetric: only Q was pre-normed).
- Symmetric gated fusion: graph_info AND set_info each have their own learned
  gate. Replaces the ln_graph_info workaround with a LayerScale warm-start.
- SwiGLU (Shazeer 2020) optional and on by default for local + set FFNs.

Equivariance contract (unchanged)
---------------------------------
Set-level operations route information into SCALAR features only. Vector
features are touched only by PaiNN. LayerScale on vectors uses a per-channel
multiplier (gamma[F]) which preserves rotation equivariance. The
`node_set_routing="both"` flag remains as an ablation lever that intentionally
breaks equivariance (set-dependent magnitude scaling on vectors).

Back-compat
-----------
Forward signatures unchanged. New constructor kwargs default to the modernized
behavior. To approximate the previous block: pass
    local_ffn=False, swiglu=False, layer_scale_init=0.0, pooling="attn"
Old checkpoints will NOT load directly (gate output dim doubled, ln_graph_info
removed, new LayerScale params, optional local FFN).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

# torch_scatter is the recommended C++/CUDA backend; we fall back to a
# pure-torch equivalent when unavailable.
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
# Generic helpers
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
    """Stochastic depth, applied per leading-dim sample (per-atom or per-graph row)."""

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


class LayerScale(nn.Module):
    """CaiT-style learnable per-channel residual scale; small init for warm start.

    Always broadcasts on the last (feature) dim. Used only on scalar features
    in this module -- both flat [N, F] and batched [S, K, F]. Vector features
    are not modulated by LayerScale here (would require a separate per-channel
    multiplier on the feature dim of [N, F, 3]); we deliberately keep
    LayerScale off the vector path so equivariance is trivially preserved.
    """

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


def _act(name: str = "silu") -> nn.Module:
    return {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
    }.get(name, nn.SiLU)()


# =============================================================================
# FFN variants (scalar features only)
# =============================================================================
class _SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer 2020).  silu(W1 x) * (W2 x) -> W3."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.fc3 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc12(x).chunk(2, dim=-1)
        return self.drop(self.fc3(F.silu(gate) * value))


class _MLP_FFN(nn.Module):
    """Vanilla transformer FFN: Linear -> SiLU -> drop -> Linear -> drop."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_ffn(dim: int, hidden_dim: int, dropout: float, swiglu: bool) -> nn.Module:
    if swiglu:
        return _SwiGLU(dim, hidden_dim, dropout=dropout)
    return _MLP_FFN(dim, hidden_dim, dropout=dropout)


# =============================================================================
# PaiNN per-graph core (one block) -- intentionally unchanged
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
        rij_hat = rij / r.unsqueeze(-1)
        env = self.rbf(r) * cosine_cutoff(r, self.cutoff).unsqueeze(-1)
        f = self.filter_net(env)  # [E, 3F]
        m = self.phi(s[src]) * f  # [E, 3F]
        m_s, m_vv, m_vd = m.split(self.dim, dim=-1)

        ds = scatter(m_s, dst, dim=0, dim_size=s.size(0), reduce="add")
        v_src = v[src]
        msg_v = m_vv.unsqueeze(-1) * v_src + m_vd.unsqueeze(-1) * rij_hat.unsqueeze(1)
        dv = scatter(msg_v, dst, dim=0, dim_size=v.size(0), reduce="add")

        s = s + self.dropout(ds)
        v = v + dv

        # Gated equivariant update.
        Uv = self.U(v.transpose(-1, -2)).transpose(-1, -2)  # [N, F, 3]
        Vv = self.V(v.transpose(-1, -2)).transpose(-1, -2)  # [N, F, 3]
        Vv_norm = Vv.norm(dim=-1)  # [N, F]
        upd_input = torch.cat([s, Vv_norm], dim=-1)
        a = self.update_net(upd_input)
        a_vv, a_sv, a_ss = a.split(self.dim, dim=-1)
        v = v + a_vv.unsqueeze(-1) * Uv
        UV_dot = (Uv * Vv).sum(dim=-1)
        s = s + a_ss + a_sv * UV_dot
        return s, v


# =============================================================================
# Pooling-by-Multihead-Attention (Set Transformer; Lee et al. 2019)
# =============================================================================
class PMA(nn.Module):
    """Pool a variable-size atom set per graph via cross-attention from
    learned seeds. Strictly more expressive than the scalar-gate-attn pool;
    with num_seeds=1 it produces one token per graph.
    """

    def __init__(
        self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0
    ):
        super().__init__()
        self.num_seeds = num_seeds
        self.seeds = nn.Parameter(torch.randn(num_seeds, dim) * 0.02)
        self.ln_kv = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.merge = nn.Linear(num_seeds * dim, dim) if num_seeds > 1 else nn.Identity()

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x_dense, mask = to_dense_batch(x, batch)
        x_dense = self.ln_kv(x_dense)
        G = x_dense.size(0)
        q = self.seeds.unsqueeze(0).expand(G, -1, -1)
        out, _ = self.mha(
            q, x_dense, x_dense, key_padding_mask=~mask, need_weights=False
        )
        if self.num_seeds == 1:
            return out.squeeze(1)
        return self.merge(out.flatten(1))


# =============================================================================
# Per-atom multi-token cross-attention (kept; pre-LN on KV done at call site)
# =============================================================================
class PerNodeMultiTokenCrossAttention(nn.Module):
    """Per-atom cross-attention to the K post-set-attention tokens of the
    atom's home graph.  K=1 is the default use here."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
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
    Vectors are not pooled -- they cannot be aggregated across atoms while
    preserving equivariance unless a frame is chosen."""
    return scatter(s, batch, dim=0, reduce=reduce)


# =============================================================================
# Shared scaffolding for the two GSC variants
# =============================================================================
class _GraphSetConvPaiNNBase(nn.Module):
    """Shared scaffolding: PaiNN -> [local FFN] -> per-graph pool ->
    set transformer -> [per-atom set context (overridden)] -> gated fusion.

    All set-level operations act on SCALAR features only. Vectors are
    touched only by PaiNN, except in the `node_set_routing="both"` ablation,
    which applies a learned set-dependent multiplicative gate to vectors
    (intentionally breaks equivariance).
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
        # ---- new knobs ----
        local_ffn: bool = True,
        swiglu: bool = True,
        pooling: str = "attn",
        pool_heads: Optional[int] = None,
        pool_seeds: int = 1,
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()
        assert node_set_routing in ("scalar", "both"), node_set_routing
        self.dim = dim
        self.node_set_routing = node_set_routing
        self.use_gating = use_gating
        self.local_ffn = local_ffn
        self.pooling = pooling

        if num_heads == 0:
            num_heads = max(1, dim // 16)
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self._mhsa_dropout = mhsa_dropout
        self._drop_path_prob = drop_path
        self._layer_scale_init = layer_scale_init

        ph = pool_heads if pool_heads is not None else num_heads
        ffn_hidden = int(dim * ffn_multiplier)

        def _ls() -> nn.Module:
            return (
                LayerScale(dim, layer_scale_init)
                if layer_scale_init > 0
                else nn.Identity()
            )

        self._ls_factory = _ls  # subclasses use this for their own branches

        # ---- 1. PaiNN per-graph propagation ----
        self.painn = PaiNNCore(dim, num_rbf, cutoff, dropout=ffn_dropout)

        # ---- 2. Optional per-atom scalar FFN (residual; scalar-only -> equivariance-safe) ----
        if local_ffn:
            self.ln_local = nn.LayerNorm(dim)
            self.local_ffn_block = _make_ffn(dim, ffn_hidden, ffn_dropout, swiglu)
            self.drop_path_local = DropPath(drop_path)
            self.ls_local = _ls()

        # ---- 3. Per-graph pool over atoms ----
        if pooling == "attn":
            # Single-token gate-attention pool (back-compat default).
            self.gate_pool = nn.Sequential(
                nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
            )
        elif pooling == "pma":
            self.pma = PMA(dim, ph, num_seeds=pool_seeds, dropout=mhsa_dropout)
        else:
            raise ValueError(f"Unknown pooling: {pooling!r}. Use 'attn' or 'pma'.")

        # ---- 4. Set-level transformer (pre-LN MHSA + FFN, both with LS+DropPath) ----
        self.ln1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            dim, num_heads, dropout=mhsa_dropout, batch_first=True
        )
        self.drop_path_attn = DropPath(drop_path)
        self.ls_attn = _ls()

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = _make_ffn(dim, ffn_hidden, ffn_dropout, swiglu)
        self.drop_path_ffn = DropPath(drop_path)
        self.ls_ffn = _ls()

        # NOTE: deliberately no ln_post on z_dense -- post-norming the per-graph
        # tokens before broadcasting strips inter-graph statistics that
        # downstream layers might use. The per-residual LayerScale + sigmoid
        # gates already control magnitude growth in the per-atom residual stream.

        # ---- 5. Symmetric gated fusion (replaces ln_graph_info workaround) ----
        # Both graph_info and set_info contributions are gated *and* LayerScale'd.
        # At init: sigmoid(0) * LS(1e-4) * x ~ 5e-5 * x per branch -- the block
        # starts as PaiNN-only and learns to use set context. No magnitude
        # doubling, no need for a per-branch LayerNorm.
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(3 * dim, dim),
                nn.SiLU(),
                nn.Linear(dim, 2 * dim),
            )
            nn.init.zeros_(self.gate[-1].bias)
        self.ls_graph = _ls()
        self.ls_set = _ls()

        # ---- 6. Optional vector-channel ablation (breaks equivariance) ----
        if node_set_routing == "both":
            self.vec_gate = nn.Linear(dim, dim, bias=True)
            nn.init.constant_(self.vec_gate.bias, -1.0)

    # -----------------------------------------------------------------------
    def _pool(self, s: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Atoms -> one token per graph."""
        if self.pooling == "attn":
            gate_logits = self.gate_pool(s)  # [N, 1]
            s_dense, mask = to_dense_batch(s, batch)  # [G, Nmax, D]
            g_dense, _ = to_dense_batch(gate_logits, batch)  # [G, Nmax, 1]
            g_dense = g_dense.masked_fill(~mask.unsqueeze(-1), -1e9)
            w = F.softmax(g_dense, dim=1)
            return (w * s_dense).sum(dim=1)
        return self.pma(s, batch)

    # -----------------------------------------------------------------------
    def _set_transformer(self, z_graph: torch.Tensor, set_batch: torch.Tensor):
        """Set-level MHSA + FFN over per-graph tokens belonging to the same set."""
        z_dense, mask = to_dense_batch(z_graph, set_batch)
        zn = self.ln1(z_dense)
        z_attn, _ = self.mha(zn, zn, zn, key_padding_mask=~mask, need_weights=False)
        z_dense = z_dense + self.drop_path_attn(self.ls_attn(z_attn))
        z_dense = z_dense + self.drop_path_ffn(self.ls_ffn(self.ffn(self.ln2(z_dense))))
        z_set_per_graph = z_dense[mask]
        return z_dense, mask, z_set_per_graph

    # -----------------------------------------------------------------------
    def _compute_set_info(
        self,
        s: torch.Tensor,
        z_dense: torch.Tensor,
        mask: torch.Tensor,
        batch: torch.Tensor,
        set_batch: torch.Tensor,
        z_set_per_graph: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    # -----------------------------------------------------------------------
    def _fuse(
        self,
        s: torch.Tensor,
        graph_info: torch.Tensor,
        set_info: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric gated fusion of graph_info and set_info into the
        per-atom scalar residual stream."""
        if self.use_gating:
            g = self.gate(torch.cat([s, graph_info, set_info], dim=-1))
            g_graph, g_set = g.chunk(2, dim=-1)
            return (
                s
                + self.ls_graph(torch.sigmoid(g_graph) * graph_info)
                + self.ls_set(torch.sigmoid(g_set) * set_info)
            )
        return s + self.ls_graph(graph_info) + self.ls_set(set_info)

    # -----------------------------------------------------------------------
    def forward(self, s, v, pos, edge_index, batch, set_batch):
        # 1. PaiNN propagation (its own internal residuals; we don't muck with them).
        s, v = self.painn(s, v, pos, edge_index)

        # 2. Optional per-atom scalar FFN (residual; scalar-only -> equivariance-safe).
        if self.local_ffn:
            s = s + self.drop_path_local(
                self.ls_local(self.local_ffn_block(self.ln_local(s)))
            )

        # 3. Per-graph scalar pool: atoms -> one token per graph.
        z_graph = self._pool(s, batch)

        # 4. Set transformer over per-graph tokens.
        z_dense, mask, z_set_per_graph = self._set_transformer(z_graph, set_batch)

        # 5. Per-atom set context (subclass-specific).
        graph_info = z_graph[batch]  # pre-set-attn token
        set_info = self._compute_set_info(
            s, z_dense, mask, batch, set_batch, z_set_per_graph
        )

        # 6. Symmetric gated fusion (LayerScale gives a warm-cold start).
        s = self._fuse(s, graph_info, set_info)

        # 7. Optional vector ablation (breaks equivariance).
        if self.node_set_routing == "both":
            vg = torch.sigmoid(self.vec_gate(set_info))  # [N, F]
            v = v * (1.0 + vg).unsqueeze(-1)

        return s, v


# =============================================================================
# Broadcast variant
# =============================================================================
class GraphSetConvPaiNNBroadcast(_GraphSetConvPaiNNBase):
    """Each atom receives the post-set-attention token of its home graph,
    broadcast uniformly across atoms in that graph."""

    def _compute_set_info(self, s, z_dense, mask, batch, set_batch, z_set_per_graph):
        return z_set_per_graph[batch]


# =============================================================================
# CrossAttn variant
# =============================================================================
class GraphSetConvPaiNNCrossAttn(_GraphSetConvPaiNNBase):
    """Each atom cross-attends to the post-set-attention token of its home
    graph (K=1 multi-token CA). Pre-LN on Q (`ln_q`) and on KV (`ln_kv`)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim, num_heads = self.dim, self.num_heads
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.node_ca = PerNodeMultiTokenCrossAttention(
            dim, num_heads, dropout=self._mhsa_dropout, qk_norm=False
        )
        self.drop_path_ca = DropPath(self._drop_path_prob)
        self.ls_ca = self._ls_factory()

    def _compute_set_info(self, s, z_dense, mask, batch, set_batch, z_set_per_graph):
        # K=1 case: each graph contributes one post-set-attn token.
        graph_tokens = z_set_per_graph.unsqueeze(1)  # [G, 1, D]
        ca_out = self.node_ca(self.ln_q(s), self.ln_kv(graph_tokens), batch)
        return self.drop_path_ca(self.ls_ca(ca_out))


# =============================================================================
# Pure-pipeline baselines (UNCHANGED below this line).
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
        z_dense, mask = to_dense_batch(z_graph, set_batch)
        lw_dense, _ = to_dense_batch(log_boltz_w.unsqueeze(-1), set_batch)
        lw_dense = lw_dense.squeeze(-1)
        lw_masked = lw_dense.masked_fill(~mask, float("-inf"))
        w = torch.softmax(lw_masked, dim=1).unsqueeze(-1)
        z_set = (w * z_dense).sum(dim=1)
        return self.proj(z_set)


class AttentionalSetPool(nn.Module):
    """Learned attention pool over conformers: graph -> set.
    Per-conformer scalar gate, softmax across conformers in a set."""

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
        # log_boltz_w accepted for API uniformity; not used.
        gate_logits = self.gate(z_graph)
        gates_dense, mask = to_dense_batch(gate_logits, set_batch)
        z_dense, _ = to_dense_batch(z_graph, set_batch)
        gates_dense = gates_dense.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(gates_dense, dim=1)
        z_set = (w * z_dense).sum(dim=1)
        return self.proj(z_set)


class EnergyFeature(nn.Module):
    """Lift a scalar log-Boltzmann-weight to a `dim`-vector via Fourier
    expansion + linear, suitable for adding to z_graph as a per-conformer
    feature."""

    def __init__(self, dim: int, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", torch.pi * (2.0**freqs), persistent=False)
        self.proj = nn.Linear(2 * num_freqs + 1, dim)

    def forward(self, log_boltz_w: torch.Tensor) -> torch.Tensor:
        x = log_boltz_w.unsqueeze(-1)
        sinusoids = torch.cat(
            [torch.sin(x * self.freqs), torch.cos(x * self.freqs)], dim=-1
        )
        feats = torch.cat([x, sinusoids], dim=-1)
        return self.proj(feats)
