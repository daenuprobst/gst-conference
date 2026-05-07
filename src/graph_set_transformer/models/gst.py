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


class DropPath(nn.Module):
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


class PerNodeSetCrossAttention(nn.Module):
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
        k_set = k_set.view(S, K_max, H, Hd).transpose(1, 2)
        v_set = v_set.view(S, K_max, H, Hd).transpose(1, 2)
        k = k_set[node_set_id]
        v = v_set[node_set_id]
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


class GraphSetConv(nn.Module):
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
    ):
        super().__init__()
        self.filters = filters
        self.pooling = pooling
        self.use_gating = use_gating
        self.node_set_mode = node_set_mode

        self.gcn_layer = GCNConv(in_channels, filters, improved=True)
        self.gcn_norm = GraphNorm(filters)
        self.gcn_dropout = nn.Dropout(
            gcn_dropout if gcn_dropout is not None else ffn_dropout
        )
        self.act = self._build_activation(activation)

        if pooling == "attn":
            self.attn_pooling = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(filters, filters),
                    nn.ReLU(),
                    nn.Linear(filters, 1),
                ),
                nn=None,
            )
        elif pooling == "multi":
            self.pool_proj = nn.Linear(2 * filters, filters)

        if num_heads == 0:
            num_heads = max(1, filters // 16)
        assert filters % num_heads == 0, (
            f"filters ({filters}) must be divisible by num_heads ({num_heads})"
        )

        self.ln1 = nn.LayerNorm(filters)
        self.mha = nn.MultiheadAttention(
            embed_dim=filters,
            num_heads=num_heads,
            dropout=mhsa_dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(filters)
        self.ffn = nn.Sequential(
            nn.Linear(filters, filters * ffn_multiplier),
            self._build_activation(activation),
            nn.Dropout(ffn_dropout),
            nn.Linear(filters * ffn_multiplier, filters),
            nn.Dropout(ffn_dropout),
        )

        self.drop_path_attn = DropPath(drop_path)
        self.drop_path_ffn = DropPath(drop_path)

        if node_set_mode == "cross_attn":
            self.ln_q = nn.LayerNorm(filters)
            self.node_ca = PerNodeSetCrossAttention(
                filters,
                num_heads,
                dropout=mhsa_dropout,
            )
        elif node_set_mode != "broadcast":
            raise ValueError(
                f"Unknown node_set_mode: {node_set_mode!r}. "
                "Use 'cross_attn' or 'broadcast'."
            )

        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(filters * 2, filters),
                self._build_activation(activation),
                nn.Linear(filters, filters),
            )
            nn.init.constant_(self.gate[-1].bias, -1.0)

    def _build_activation(self, activation):
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        return activations.get(activation, nn.ReLU)()

    def _pool_graphs(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        elif self.pooling == "sum":
            return global_add_pool(x, batch)
        elif self.pooling == "multi":
            pooled = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1
            )
            return self.pool_proj(pooled)
        elif self.pooling == "attn":
            return self.attn_pooling(x, batch)
        else:
            return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch, set_batch, return_graph_emb=False):
        h = self.gcn_layer(x, edge_index)
        h = self.gcn_norm(h, batch)
        h = self.act(h)
        h = self.gcn_dropout(h)
        x = x + h

        z_graph = self._pool_graphs(x, batch)

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
        z_dense = z_dense + self.drop_path_attn(z_attn)
        z_dense = z_dense + self.drop_path_ffn(self.ffn(self.ln2(z_dense)))
        z_set = z_dense[mask]

        graph_info = z_graph[batch]
        if self.node_set_mode == "cross_attn":
            node_set_id = set_batch[batch]
            set_info = self.node_ca(
                self.ln_q(x),
                z_dense,
                node_set_id,
                mask,
            )
        else:
            set_info = z_set[batch]

        if self.use_gating:
            gate_input = torch.cat([x, set_info], dim=-1)
            g_set = torch.sigmoid(self.gate(gate_input))
            x_out = x + graph_info + g_set * set_info
        else:
            x_out = x + graph_info + set_info

        if return_graph_emb:
            return x_out, z_set

        return x_out
