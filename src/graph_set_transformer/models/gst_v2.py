"""
Hierarchical graph-set transformer block (redesigned).

Forward signature is unchanged from the prior version:
    forward(x, edge_index, batch, set_batch, return_graph_emb=False)
        -> x_out                  # if not return_graph_emb
        -> (x_out, z_set)         # otherwise

Pipeline (per call)
-------------------
1. Input projection                                         (NEW; fixes residual when in_channels != filters)
2. Local MPNN: GCN -> GraphNorm -> act -> dropout           (residual + DropPath + LayerScale)
3. Local FFN: pre-LN + MLP                           (NEW; residual + DropPath + LayerScale)
4. Per-graph pool: mean | max | sum | multi | attn | pma    ("pma" is NEW)
5. Set transformer: pre-LN MHSA + pre-LN FFN over graphs in the same set
6. Per-node set context: cross-attn from each node to its set's graph tokens
   (pre-LN on Q and KV; LayerScale on output) -- or "broadcast"
7. Gated fusion of x, graph_info, set_info                  (NEW: symmetric, sees all three)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    AttentionalAggregation,
    GCNConv,
    GraphNorm,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth, applied per leading-dim sample (per-node or per-set row)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


class LayerScale(nn.Module):
    """CaiT-style learnable per-channel residual scale; small init for warm start."""

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x):
        return x * self.gamma


def _build_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    return activations.get(name, nn.ReLU)()


class MLP_FFN(nn.Module):
    """Vanilla transformer FFN: Linear -> act -> drop -> Linear -> drop."""

    def __init__(
        self, dim: int, hidden_dim: int, activation: str = "silu", dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def _make_ffn(dim: int, hidden_dim: int, activation: str, dropout: float) -> nn.Module:
    return MLP_FFN(dim, hidden_dim, activation=activation, dropout=dropout)


# ---------------------------------------------------------------------------
# Pooling-by-Multihead-Attention (Set Transformer; Lee et al. 2019)
# ---------------------------------------------------------------------------
class PMA(nn.Module):
    """Pool a variable-size set of vectors via cross-attention from learned seeds.

    Strictly more expressive than `AttentionalAggregation` (which uses a single
    scalar gate). With num_seeds=1 it produces a single pooled vector per graph.
    """

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1):
        super().__init__()
        self.num_seeds = num_seeds
        self.seeds = nn.Parameter(torch.randn(num_seeds, dim) * 0.02)
        self.ln_kv = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.merge = nn.Linear(num_seeds * dim, dim) if num_seeds > 1 else nn.Identity()

    def forward(self, x, batch):
        x_dense, mask = to_dense_batch(x, batch)  # [G, Nmax, D], [G, Nmax]
        x_dense = self.ln_kv(x_dense)
        G = x_dense.size(0)
        q = self.seeds.unsqueeze(0).expand(G, -1, -1)  # [G, S, D]
        out, _ = self.mha(
            q, x_dense, x_dense, key_padding_mask=~mask, need_weights=False
        )
        if self.num_seeds == 1:
            return out.squeeze(1)  # [G, D]
        return self.merge(out.flatten(1))  # [G, D]


# ---------------------------------------------------------------------------
# Per-node cross-attention to set tokens (unchanged core; same SDPA path)
# ---------------------------------------------------------------------------
class PerNodeSetCrossAttention(nn.Module):
    """Each node attends to the K_max graph-level tokens in its set."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x_nodes, z_set_tokens, node_set_id, set_mask):
        N = x_nodes.shape[0]
        S, K_max, D = z_set_tokens.shape
        H, Hd = self.num_heads, self.head_dim

        q = self.q_proj(x_nodes).view(N, H, 1, Hd)
        kv = self.kv_proj(z_set_tokens)
        k_set, v_set = kv.chunk(2, dim=-1)
        k_set = k_set.view(S, K_max, H, Hd).transpose(1, 2)  # [S, H, K_max, Hd]
        v_set = v_set.view(S, K_max, H, Hd).transpose(1, 2)
        k = k_set[node_set_id]  # [N, H, K_max, Hd]
        v = v_set[node_set_id]
        # SDPA convention: attn_mask True == attend (the set_mask passed in is
        # already True-for-valid, False-for-padding).
        attn_mask = set_mask[node_set_id].view(N, 1, 1, K_max)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(N, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------
class GraphSetConv(nn.Module):
    """Hierarchical graph-set transformer block.

    Designed to be stackable: every residual branch has its own DropPath and
    LayerScale, so multiple of these in series train stably without tuning.
    """

    def __init__(
        self,
        filters,
        in_channels=3,
        activation="silu",
        mhsa_dropout=0.0,
        ffn_dropout=0.0,
        gcn_dropout=None,
        pooling="attn",
        use_gating=True,
        ffn_multiplier=2,
        num_heads=4,
        drop_path=0.0,
        node_set_mode="cross_attn",
        # ---- new knobs (all back-compat by default) ----
        local_ffn: bool = True,
        pool_heads: int | None = None,
        pool_seeds: int = 1,
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()
        self.filters = filters
        self.pooling = pooling
        self.use_gating = use_gating
        self.node_set_mode = node_set_mode
        self.local_ffn = local_ffn

        # Resolve heads.
        if num_heads == 0:
            num_heads = max(1, filters // 16)
        assert filters % num_heads == 0, (
            f"filters ({filters}) must be divisible by num_heads ({num_heads})"
        )

        def _ls() -> nn.Module:
            return (
                LayerScale(filters, layer_scale_init)
                if layer_scale_init > 0
                else nn.Identity()
            )

        # -- Input projection (fixes residual when in_channels != filters) --
        self.input_proj = (
            nn.Linear(in_channels, filters) if in_channels != filters else nn.Identity()
        )

        # -- 1. Local MPNN --
        self.gcn_layer = GCNConv(filters, filters, improved=True)
        self.gcn_norm = GraphNorm(filters)
        self.gcn_dropout = nn.Dropout(
            gcn_dropout if gcn_dropout is not None else ffn_dropout
        )
        self.act = _build_activation(activation)
        self.drop_path_mpnn = DropPath(drop_path)
        self.ls_mpnn = _ls()

        # -- 2. Local FFN (per-node, pre-LN) --
        if local_ffn:
            self.ln_local = nn.LayerNorm(filters)
            self.local_ffn_block = _make_ffn(
                filters, filters * ffn_multiplier, activation, ffn_dropout
            )
            self.drop_path_local = DropPath(drop_path)
            self.ls_local = _ls()

        # -- 3. Per-graph pool --
        ph = pool_heads if pool_heads is not None else num_heads
        if pooling == "attn":
            self.attn_pooling = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(filters, filters),
                    _build_activation(activation),
                    nn.Linear(filters, 1),
                ),
                nn=None,
            )
        elif pooling == "multi":
            self.pool_proj = nn.Linear(2 * filters, filters)
        elif pooling == "pma":
            self.pma = PMA(filters, ph, num_seeds=pool_seeds)
        elif pooling not in {"mean", "max", "sum"}:
            raise ValueError(
                f"Unknown pooling: {pooling!r}. "
                "Use one of: mean, max, sum, multi, attn, pma."
            )

        # -- 4. Set transformer (pre-LN MHSA + FFN) --
        self.ln1 = nn.LayerNorm(filters)
        self.mha = nn.MultiheadAttention(
            embed_dim=filters,
            num_heads=num_heads,
            dropout=mhsa_dropout,
            batch_first=True,
        )
        self.drop_path_attn = DropPath(drop_path)
        self.ls_attn = _ls()

        self.ln2 = nn.LayerNorm(filters)
        self.ffn = _make_ffn(filters, filters * ffn_multiplier, activation, ffn_dropout)
        self.drop_path_ffn = DropPath(drop_path)
        self.ls_ffn = _ls()

        # -- 5. Per-node set context --
        if node_set_mode == "cross_attn":
            self.ln_q = nn.LayerNorm(filters)
            self.ln_kv = nn.LayerNorm(filters)
            self.node_ca = PerNodeSetCrossAttention(
                filters, num_heads, dropout=mhsa_dropout
            )
            self.drop_path_ca = DropPath(drop_path)
            self.ls_ca = _ls()
        elif node_set_mode != "broadcast":
            raise ValueError(
                f"Unknown node_set_mode: {node_set_mode!r}. Use 'cross_attn' or 'broadcast'."
            )

        # -- 6. Gated fusion (symmetric: separate gates for graph_info, set_info) --
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(filters * 3, filters),
                _build_activation(activation),
                nn.Linear(filters, filters * 2),
            )
            # Neutral start: sigmoid(0) = 0.5. The cross-attn LayerScale already
            # supplies the warm-start; gate doesn't need a -1 bias hack.
            nn.init.zeros_(self.gate[-1].bias)

    # -----------------------------------------------------------------------
    def _pool_graphs(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.pooling == "max":
            return global_max_pool(x, batch)
        if self.pooling == "sum":
            return global_add_pool(x, batch)
        if self.pooling == "multi":
            pooled = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1
            )
            return self.pool_proj(pooled)
        if self.pooling == "attn":
            return self.attn_pooling(x, batch)
        if self.pooling == "pma":
            return self.pma(x, batch)
        return global_mean_pool(x, batch)

    # -----------------------------------------------------------------------
    def forward(self, x, edge_index, batch, set_batch, return_graph_emb=False):
        # 0. Input projection (no-op when in_channels == filters)
        x = self.input_proj(x)

        # 1. Local MPNN with residual
        h = self.gcn_layer(x, edge_index)
        h = self.gcn_norm(h, batch)
        h = self.act(h)
        h = self.gcn_dropout(h)
        x = x + self.drop_path_mpnn(self.ls_mpnn(h))

        # 2. Local FFN with residual (per-node MLP, pre-LN)
        if self.local_ffn:
            x = x + self.drop_path_local(
                self.ls_local(self.local_ffn_block(self.ln_local(x)))
            )

        # 3. Per-graph pool
        z_graph = self._pool_graphs(x, batch)

        # 4. Set transformer (pre-LN MHSA + pre-LN FFN over graphs in same set)
        z_dense, mask = to_dense_batch(z_graph, set_batch)
        mask = mask.bool()
        z_normed = self.ln1(z_dense)
        z_attn, _ = self.mha(
            z_normed,
            z_normed,
            z_normed,
            key_padding_mask=~mask,
            need_weights=False,
        )
        z_dense = z_dense + self.drop_path_attn(self.ls_attn(z_attn))
        z_dense = z_dense + self.drop_path_ffn(self.ls_ffn(self.ffn(self.ln2(z_dense))))
        z_set = z_dense[mask]  # [num_graphs, D]

        # 5. Per-node set context
        graph_info = z_graph[batch]  # pre-set-attn graph token
        if self.node_set_mode == "cross_attn":
            node_set_id = set_batch[batch]
            ca_out = self.node_ca(
                self.ln_q(x),
                self.ln_kv(z_dense),  # pre-LN on KV (was missing)
                node_set_id,
                mask,
            )
            set_info = self.drop_path_ca(self.ls_ca(ca_out))
        else:  # "broadcast": each node receives its graph's set-attended token
            set_info = z_set[batch]

        # 6. Symmetric gated fusion
        if self.use_gating:
            gate_input = torch.cat([x, graph_info, set_info], dim=-1)
            g_graph, g_set = self.gate(gate_input).chunk(2, dim=-1)
            x_out = (
                x
                + torch.sigmoid(g_graph) * graph_info
                + torch.sigmoid(g_set) * set_info
            )
        else:
            x_out = x + graph_info + set_info

        if return_graph_emb:
            return x_out, z_set
        return x_out
