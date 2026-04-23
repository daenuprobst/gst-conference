import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset, MoleculeNet
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    aggr,
    GraphNorm,
)
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import scatter
from torch_geometric.utils import to_dense_batch
from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*torch-scatter.*",
    category=UserWarning,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Wrappers to replace torch_scatter functions
def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


def scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")


class SetDataset(Dataset):
    def __init__(self, sets):
        self.sets = sets

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        return self.sets[idx]


def collate_sets(batch_of_sets):
    all_graphs = []
    set_assignments = []
    labels = []

    for set_idx, (graph_set, label) in enumerate(batch_of_sets):
        all_graphs.extend(graph_set)
        set_assignments.extend([set_idx] * len(graph_set))
        labels.append(label)

    return (
        Batch.from_data_list(all_graphs),
        torch.tensor(set_assignments, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# Set Transformer (original implimentation and a bit of hackiness to add masking)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, key_padding_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)

        if key_padding_mask is not None:
            batch_size = key_padding_mask.size(0)

            mask = key_padding_mask.unsqueeze(1)
            mask = mask.repeat(self.num_heads, A.size(1), 1)

            A = A.masked_fill(mask, float("-inf"))

        A = torch.softmax(A, 2)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, key_padding_mask=None):
        return self.mab(X, X, key_padding_mask=key_padding_mask)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, key_padding_mask=None):
        H = self.mab0(
            self.I.repeat(X.size(0), 1, 1), X, key_padding_mask=key_padding_mask
        )

        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, key_padding_mask=None):
        return self.mab(
            self.S.repeat(X.size(0), 1, 1), X, key_padding_mask=key_padding_mask
        )


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input,
        num_outputs,
        dim_output,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.ModuleList(
            [
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ]
        )
        self.dec = nn.ModuleList(
            [
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            ]
        )
        self.fc_out = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, key_padding_mask=None):
        # Encoder
        for layer in self.enc:
            X = layer(X, key_padding_mask=key_padding_mask)

        # Decoder
        X = self.dec[0](X, key_padding_mask=key_padding_mask)  # PMA
        X = self.dec[1](X)  # SAB
        X = self.dec[2](X)  # SAB
        X = self.fc_out(X)

        return X


class SetTransformerGraphClassifier(nn.Module):
    def __init__(
        self, in_channels, hidden_dim, num_classes, num_heads=4, num_sabs=2, dropout=0.1
    ):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Add dropout layer

        self.set_transformer = SetTransformer(hidden_dim, 1, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, set_batch):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)  # Apply dropout
        x = self.act(self.conv2(x, edge_index))
        x = self.dropout(x)  # Apply dropout

        graph_emb = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

        z_padded, key_padding_mask = self._pad_to_sets(graph_emb, set_batch)

        set_emb = self.set_transformer(z_padded, key_padding_mask)

        set_emb = set_emb.squeeze(1)

        return self.classifier(set_emb)

    def _pad_to_sets(self, graph_emb, set_batch):
        num_graphs = graph_emb.size(0)
        hidden_dim = graph_emb.size(1)
        device = graph_emb.device

        num_sets = int(set_batch.max()) + 1
        set_sizes = scatter_add(
            torch.ones_like(set_batch), set_batch, dim=0, dim_size=num_sets
        )
        max_set_size = int(set_sizes.max())

        sorted_indices = torch.argsort(set_batch)
        sorted_set_batch = set_batch[sorted_indices]

        ones = torch.ones(num_graphs, dtype=torch.long, device=device)
        cumsum = torch.cumsum(ones, dim=0)
        set_offsets = torch.zeros(num_sets + 1, dtype=torch.long, device=device)
        set_offsets[1:] = torch.cumsum(set_sizes, dim=0)
        positions_sorted = cumsum - 1 - set_offsets[sorted_set_batch]

        positions = torch.empty_like(positions_sorted)
        positions[sorted_indices] = positions_sorted

        # Padding
        z_padded = torch.zeros(num_sets, max_set_size, hidden_dim, device=device)
        z_padded[set_batch, positions] = graph_emb

        key_padding_mask = torch.ones(
            num_sets, max_set_size, dtype=torch.bool, device=device
        )
        key_padding_mask[set_batch, positions] = False

        return z_padded, key_padding_mask


# DeepSets (adapted from the barebones original implementation)


class DeepSets(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, aggregator="sum", dropout=0.0
    ):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        if aggregator == "max":
            self.aggregator = aggr.MaxAggregation()
        elif aggregator == "mean":
            self.aggregator = aggr.MeanAggregation()
        elif aggregator == "sum":
            self.aggregator = aggr.SumAggregation()
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        h = self.psi(x)
        h = self.aggregator(h, dim=1).squeeze(1)
        y = self.phi(h)

        return y


class DeepSetGraphClassifier(nn.Module):
    def __init__(
        self, in_channels, hidden_dim, num_classes, aggregator="sum", dropout=0.1
    ):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Add dropout layer

        self.deepsets = DeepSets(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            aggregator=aggregator,
            dropout=dropout,
        )

    def forward(self, data, set_batch):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)  # Apply dropout
        x = self.act(self.conv2(x, edge_index))
        x = self.dropout(x)  # Apply dropout
        x = self.act(self.conv3(x, edge_index))
        x = self.dropout(x)  # Apply dropout

        graph_emb = global_mean_pool(x, batch)
        x_padded = self._pad_to_sets(graph_emb, set_batch)

        return self.deepsets(x_padded)

    def _pad_to_sets(self, graph_emb, set_batch):
        num_graphs = graph_emb.size(0)
        hidden_dim = graph_emb.size(1)
        device = graph_emb.device

        num_sets = int(set_batch.max()) + 1
        set_sizes = scatter_add(
            torch.ones_like(set_batch), set_batch, dim=0, dim_size=num_sets
        )
        max_set_size = int(set_sizes.max())

        # Compute positions within each set
        sorted_indices = torch.argsort(set_batch)
        sorted_set_batch = set_batch[sorted_indices]

        ones = torch.ones(num_graphs, dtype=torch.long, device=device)
        cumsum = torch.cumsum(ones, dim=0)
        set_offsets = torch.zeros(num_sets + 1, dtype=torch.long, device=device)
        set_offsets[1:] = torch.cumsum(set_sizes, dim=0)
        positions_sorted = cumsum - 1 - set_offsets[sorted_set_batch]

        positions = torch.empty_like(positions_sorted)
        positions[sorted_indices] = positions_sorted

        # Create padded tensor
        x_padded = torch.zeros(num_sets, max_set_size, hidden_dim, device=device)
        x_padded[set_batch, positions] = graph_emb

        return x_padded


# Graph set convolution (ours)
# class PerceiverPool(nn.Module):
#     def __init__(self, dim, num_queries=4, num_heads=4, dropout=0.0):
#         super().__init__()
#         self.num_queries = num_queries
#
#         self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
#
#         self.ln_q = nn.LayerNorm(dim)
#         self.ln_kv = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.ln_out = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim * 2, dim),
#         )
#
#     def forward(self, x, batch):
#         x_dense, node_mask = to_dense_batch(x, batch)
#         num_graphs = x_dense.size(0)
#
#         q = self.queries.unsqueeze(0).expand(num_graphs, -1, -1)
#         q = self.ln_q(q)
#
#         kv = self.ln_kv(x_dense)
#
#         attn_out, _ = self.attn(q, kv, kv, key_padding_mask=~node_mask)
#
#         tokens = self.ln_out(q + attn_out)
#         tokens = tokens + self.ffn(tokens)
#
#         return tokens  # [num_graphs, k, dim]
#
#
# class GraphSetConv(nn.Module):
#     def __init__(
#         self,
#         filters,
#         in_channels=3,
#         num_queries=4,
#         num_heads=4,
#         mhsa_dropout=0.2,
#         ffn_dropout=0.2,
#         use_gating=True,
#         ffn_multiplier=4,
#         activation=None,
#     ):
#         super().__init__()
#         self.filters = filters
#         self.num_queries = num_queries
#         self.use_gating = use_gating
#
#         self.gcn_layer = GCNConv(in_channels, filters, improved=True)
#         self.gcn_norms = GraphNorm(filters)
#         self.act = nn.GELU()
#         self.gcn_dropout = nn.Dropout(ffn_dropout)
#
#         # Multi-token pooling replaces _pool_graphs
#         self.pool = PerceiverPool(
#             filters, num_queries=num_queries, num_heads=num_heads, dropout=mhsa_dropout
#         )
#
#         # Set-level transformer (pre-norm)
#         self.ln1 = nn.LayerNorm(filters)
#         self.mha = nn.MultiheadAttention(
#             filters, num_heads, dropout=mhsa_dropout, batch_first=True
#         )
#         self.ln2 = nn.LayerNorm(filters)
#         self.ffn = nn.Sequential(
#             nn.Linear(filters, filters * ffn_multiplier),
#             nn.GELU(),
#             nn.Dropout(ffn_dropout),
#             nn.Linear(filters * ffn_multiplier, filters),
#             nn.Dropout(ffn_dropout),
#         )
#
#         # Broadcast: aggregate k tokens per graph back to one vector before gating
#         # (simplest option; see "richer broadcast" below for an upgrade)
#         self.token_agg = nn.Linear(filters * num_queries, filters)
#
#         if use_gating:
#             self.gate = nn.Sequential(nn.Linear(filters * 2, filters), nn.Sigmoid())
#
#     def forward(self, x, edge_index, batch, set_batch):
#         # --- Per-graph GNN encoding ---
#         x = self.gcn_layer(x, edge_index)
#         x = self.gcn_norms(x, batch)
#         x = self.act(x)
#         x = self.gcn_dropout(x)
#
#         # --- Multi-token pooling: [num_graphs, k, dim] ---
#         graph_tokens = self.pool(x, batch)  # [G, k, D]
#         G, k, D = graph_tokens.shape
#
#         # --- Build set-level token sequence ---
#         # Flatten: k tokens per graph, all sharing that graph's set_batch id
#         tokens_flat = graph_tokens.reshape(G * k, D)  # [G*k, D]
#         token_set_batch = set_batch.repeat_interleave(k)  # [G*k]
#
#         # Dense per-set tensor: [num_sets, max_tokens_in_set, D]
#         z_dense, mask = to_dense_batch(tokens_flat, token_set_batch)
#
#         # --- Pre-norm transformer block over the set ---
#         z_norm = self.ln1(z_dense)
#         z_attn, _ = self.mha(z_norm, z_norm, z_norm, key_padding_mask=~mask)
#         z_dense = z_dense + z_attn
#         z_dense = z_dense + self.ffn(self.ln2(z_dense))
#
#         # --- Unbatch back to [G*k, D] then to [G, k, D] ---
#         tokens_out = z_dense[mask]  # [G*k, D]
#         tokens_out = tokens_out.view(G, k, D)
#
#         # --- Broadcast set-aware context back to nodes ---
#         # Aggregate k tokens per graph into one summary vector per graph
#         set_info_per_graph = self.token_agg(tokens_out.reshape(G, k * D))  # [G, D]
#         set_info = set_info_per_graph[batch]  # [num_nodes, D]
#
#         if self.use_gating:
#             gate = self.gate(torch.cat([x, set_info], dim=-1))
#             x_out = x + gate * set_info
#         else:
#             x_out = x + set_info
#
#         return x_out


# class DropPath(nn.Module):
#     def __init__(self, drop_prob: float = 0.0):
#         super().__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         if self.drop_prob == 0.0 or not self.training:
#             return x
#         keep_prob = 1.0 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         mask = x.new_empty(shape).bernoulli_(keep_prob)
#         return x * mask / keep_prob
#
#
# class GraphSetConv(nn.Module):
#     def __init__(
#         self,
#         filters,
#         in_channels=3,
#         activation="silu",
#         mhsa_dropout=0.2,
#         ffn_dropout=0.2,
#         pooling="attn",
#         use_gating=True,
#         ffn_multiplier=4,
#         num_heads=4,
#         drop_path=0.1,
#     ):
#         super().__init__()
#         self.filters = filters
#         self.activation = activation
#         self.pooling = pooling
#         self.use_gating = use_gating
#
#         self.gcn_layer = GCNConv(in_channels, filters, improved=True)
#
#         self.gcn_norms = GraphNorm(filters)
#         self.gcn_dropout = nn.Dropout(ffn_dropout)
#
#         if num_heads == 0:
#             num_heads = max(1, filters // 16)
#
#         self.ln1 = nn.LayerNorm(filters)
#         self.mha = nn.MultiheadAttention(
#             embed_dim=filters,
#             num_heads=num_heads,
#             dropout=mhsa_dropout,
#             batch_first=True,
#         )
#         self.ln2 = nn.LayerNorm(filters)
#         self.ffn = nn.Sequential(
#             nn.Linear(filters, filters * ffn_multiplier),
#             self._build_activation(activation),
#             nn.Dropout(ffn_dropout),
#             nn.Linear(filters * ffn_multiplier, filters),
#             nn.Dropout(ffn_dropout),
#         )
#         self.ln_post_ffn = nn.LayerNorm(filters)
#         self.pool_proj = nn.Linear(2 * filters, filters)
#
#         self.drop_path_attn = DropPath(drop_path)
#         self.drop_path_ffn = DropPath(drop_path)
#
#         self.attn_pooling = AttentionalAggregation(
#             gate_nn=nn.Sequential(
#                 nn.Linear(filters, filters),
#                 self._build_activation("relu"),
#                 nn.Linear(filters, 1),
#             ),
#             nn=None,
#         )
#
#         if use_gating:
#             self.gate = nn.Sequential(nn.Linear(filters * 2, filters), nn.Sigmoid())
#
#         self.act = self._build_activation(activation)
#
#     def _build_activation(self, activation):
#         activations = {
#             "relu": nn.ReLU,
#             "gelu": nn.GELU,
#             "elu": nn.ELU,
#             "silu": nn.SiLU,
#             "tanh": nn.Tanh,
#         }
#         return activations.get(activation, nn.ReLU)()
#
#     def _pool_graphs(self, x, batch):
#         if self.pooling == "mean":
#             return global_mean_pool(x, batch)
#         elif self.pooling == "max":
#             return global_max_pool(x, batch)
#         elif self.pooling == "sum":
#             return global_add_pool(x, batch)
#         elif self.pooling == "multi":
#             pooled = torch.cat(
#                 [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1
#             )
#             return self.pool_proj(pooled)
#         elif self.pooling == "attn":
#             return self.attn_pooling(x, batch)
#         else:
#             return global_mean_pool(x, batch)
#
#     def forward(self, x, edge_index, batch, set_batch):
#         x = self.gcn_layer(x, edge_index)
#         x = self.gcn_norms(x, batch)
#         x = self.act(x)
#         x = self.gcn_dropout(x)
#
#         z = self._pool_graphs(x, batch)
#
#         z_dense, mask = to_dense_batch(z, set_batch)
#         mask = mask.to(dtype=torch.bool, device=z_dense.device)
#
#         # z_dense = (
#         #     z_dense
#         #     + self.mha(
#         #         self.ln1(z_dense),
#         #         self.ln1(z_dense),
#         #         self.ln1(z_dense),
#         #         key_padding_mask=~mask,
#         #     )[0]
#         # )
#         # z_dense = z_dense + self.ffn(self.ln2(z_dense))
#
#         z_attn, _ = self.mha(
#             self.ln1(z_dense),
#             self.ln1(z_dense),
#             self.ln1(z_dense),
#             key_padding_mask=~mask,
#         )
#         z_dense = z_dense + self.drop_path_attn(z_attn)
#
#         # Pre-norm FFN
#         z_dense = z_dense + self.drop_path_ffn(self.ffn(self.ln2(z_dense)))
#
#         z_out = z_dense[mask]
#
#         set_info = z_out[batch]
#
#         if self.use_gating:
#             gate_input = torch.cat([x, set_info], dim=-1)
#             gate_values = self.gate(gate_input)
#             # x_out = gate_values * set_info + (1 - gate_values) * x
#             x_out = x + gate_values * set_info
#         else:
#             x_out = x + set_info
#
#         return x_out
#
#
# class GraphSetTransformerClassifier(nn.Module):
#     def __init__(self, in_channels, hidden_dim, num_classes, dropout=0.3):
#         super().__init__()
#         self.setconv1 = GraphSetConv(
#             filters=hidden_dim,
#             in_channels=in_channels,
#             activation="silu",
#             mhsa_dropout=dropout,
#             ffn_dropout=dropout,
#         )
#         self.setconv2 = GraphSetConv(
#             filters=hidden_dim,
#             in_channels=hidden_dim,
#             activation="silu",
#             mhsa_dropout=dropout,
#             ffn_dropout=dropout,
#         )
#         self.setconv3 = GraphSetConv(
#             filters=hidden_dim,
#             in_channels=hidden_dim,
#             activation="silu",
#             mhsa_dropout=dropout,
#             ffn_dropout=dropout,
#         )
#         self.dropout = nn.Dropout(
#             dropout
#         )  # External dropout between layers (matches other models)
#         self.classifier = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, data, set_batch):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.setconv1(x, edge_index, batch, set_batch)
#         x = self.dropout(x)  # Apply dropout
#         x = self.setconv2(x, edge_index, batch, set_batch)
#         x = self.dropout(x)  # Apply dropout
#         x = self.setconv3(x, edge_index, batch, set_batch)
#         x = self.dropout(x)  # Apply dropout
#         graph_emb = global_mean_pool(x, batch)
#         set_emb = scatter_add(graph_emb, set_batch, dim=0)
#         return self.classifier(set_emb)


"""
Drop-in replacement for GraphSetConv that gives each node its *own* context
vector drawn from the set, instead of broadcasting a single graph-level vector
to every node in the graph.

Key ideas
---------
1. Each graph is represented by `num_tokens` pooled tokens (1 = original
   behavior, >1 = Perceiver-style multi-token). Tokens are produced by
   attentional pooling with `num_tokens` learned queries.
2. Set-level self-attention runs over all G * num_tokens tokens belonging
   to a set (padded + masked per set).
3. Each node CROSS-ATTENDS to the `num_tokens` tokens of its own graph.
   This is what makes the injected context node-specific.
4. A residual gate fuses node features with the node-specific context,
   then the block returns updated node features for the next GCN.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphNorm
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
        return x * mask / keep


class MultiQueryAttnPool(nn.Module):
    """Attentional pooling with `num_tokens` learned queries per graph.

    Produces [num_graphs, num_tokens, dim]. With num_tokens=1 this is
    equivalent to standard attentional aggregation.
    """

    def __init__(self, dim, num_tokens=1, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, batch):
        # x: [N, D], batch: [N]
        x_dense, node_mask = to_dense_batch(x, batch)  # [G, maxN, D], [G, maxN]
        G = x_dense.size(0)
        q = self.queries.unsqueeze(0).expand(G, -1, -1)  # [G, K, D]
        q = self.ln_q(q)
        kv = self.ln_kv(x_dense)
        # key_padding_mask: True = ignore
        out, _ = self.attn(q, kv, kv, key_padding_mask=~node_mask)
        return out  # [G, K, D]


class GraphSetConv(nn.Module):
    """Node-level set-aware conv block.

    Pipeline:
      GCN -> multi-token attn pool per graph -> set-level self-attn over all
      graphs in a set -> node cross-attn to that graph's tokens -> gated residual.
    """

    def __init__(
        self,
        filters,
        in_channels,
        num_tokens=2,
        num_heads=4,
        mhsa_dropout=0.2,
        ffn_dropout=0.2,
        ffn_multiplier=2,
        use_gating=True,
        drop_path=0.5,
        activation="silu",
    ):
        super().__init__()
        self.filters = filters
        self.num_tokens = num_tokens
        self.use_gating = use_gating

        # --- Per-graph GCN encoder ---
        self.gcn = GCNConv(in_channels, filters, improved=True)
        self.gcn_norm = GraphNorm(filters)
        self.act = self._make_act(activation)
        self.gcn_dropout = nn.Dropout(ffn_dropout)

        # --- Multi-token pooling per graph ---
        self.pool = MultiQueryAttnPool(
            dim=filters,
            num_tokens=num_tokens,
            num_heads=num_heads,
            dropout=mhsa_dropout,
        )

        # --- Set-level self-attention over (graph, token) pairs ---
        self.ln_set1 = nn.LayerNorm(filters)
        self.set_attn = nn.MultiheadAttention(
            embed_dim=filters,
            num_heads=num_heads,
            dropout=mhsa_dropout,
            batch_first=True,
        )
        self.ln_set2 = nn.LayerNorm(filters)
        self.set_ffn = nn.Sequential(
            nn.Linear(filters, filters * ffn_multiplier),
            self._make_act(activation),
            nn.Dropout(ffn_dropout),
            nn.Linear(filters * ffn_multiplier, filters),
            nn.Dropout(ffn_dropout),
        )
        self.drop_path_set_attn = DropPath(drop_path)
        self.drop_path_set_ffn = DropPath(drop_path)

        # --- Node cross-attention to own-graph tokens ---
        # Queries = nodes, Keys/Values = the K tokens of the node's graph.
        self.ln_node_q = nn.LayerNorm(filters)
        self.ln_token_kv = nn.LayerNorm(filters)
        self.node_cross_attn = nn.MultiheadAttention(
            embed_dim=filters,
            num_heads=num_heads,
            dropout=mhsa_dropout,
            batch_first=True,
        )
        self.drop_path_node = DropPath(drop_path)

        # --- Fusion gate: per-node, per-feature ---
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(2 * filters, filters),
                nn.Sigmoid(),
            )
            # Bias gate toward ~0 at init so block starts near identity.
            # sigmoid(-2) ~= 0.12; adjust if you want slower/faster ramp-up.
            nn.init.constant_(self.gate[0].bias, -2.0)

    @staticmethod
    def _make_act(name):
        return {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }.get(name, nn.ReLU)()

    def forward(self, x, edge_index, batch, set_batch):
        """
        x         : [N, in_channels]  node features
        edge_index: [2, E]
        batch     : [N]                node -> graph id
        set_batch : [G]                graph -> set id
        returns   : [N, filters]       updated node features
        """
        # ---- GCN ----
        h = self.gcn(x, edge_index)
        h = self.gcn_norm(h, batch)
        h = self.act(h)
        h = self.gcn_dropout(h)  # [N, D]

        # ---- Multi-token pooling per graph ----
        tokens = self.pool(h, batch)  # [G, K, D]
        G, K, D = tokens.shape

        # ---- Build set-level sequence and do self-attention ----
        # Flatten graph tokens; assign each token to its set id.
        tokens_flat = tokens.reshape(G * K, D)
        token_set_id = set_batch.repeat_interleave(K)  # [G*K]
        z_dense, z_mask = to_dense_batch(tokens_flat, token_set_id)
        # z_dense: [S, max_GK, D], z_mask: [S, max_GK] (True = valid)

        z_n = self.ln_set1(z_dense)
        z_attn, _ = self.set_attn(z_n, z_n, z_n, key_padding_mask=~z_mask)
        z_dense = z_dense + self.drop_path_set_attn(z_attn)
        z_dense = z_dense + self.drop_path_set_ffn(self.set_ffn(self.ln_set2(z_dense)))

        # ---- Unbatch back to per-graph token layout [G, K, D] ----
        tokens_out_flat = z_dense[z_mask]  # [G*K, D]
        tokens_out = tokens_out_flat.view(G, K, D)

        # ---- Node-specific context via cross-attention ----
        # For each node: query its own graph's K tokens.
        # Build per-node K/V by gathering tokens_out along graph dim.
        # Easiest: per node, kv = tokens_out[batch]  -> [N, K, D].
        node_q = self.ln_node_q(h).unsqueeze(1)  # [N, 1, D]
        kv = self.ln_token_kv(tokens_out[batch])  # [N, K, D]
        ctx, _ = self.node_cross_attn(node_q, kv, kv)  # [N, 1, D]
        ctx = ctx.squeeze(1)  # [N, D]
        ctx = self.drop_path_node(ctx)

        # ---- Gated residual fusion ----
        if self.use_gating:
            gate = self.gate(torch.cat([h, ctx], dim=-1))
            out = h + gate * ctx
        else:
            out = h + ctx

        return out


class GraphSetTransformerClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_classes,
        dropout=0.1,
        num_tokens=4,
        num_heads=4,
    ):
        super().__init__()
        self.setconv1 = GraphSetConv(
            hidden_dim,
            in_channels,
            num_tokens=num_tokens,
            num_heads=num_heads,
            mhsa_dropout=dropout,
            ffn_dropout=dropout,
        )
        self.setconv2 = GraphSetConv(
            hidden_dim,
            hidden_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            mhsa_dropout=dropout,
            ffn_dropout=dropout,
        )
        self.setconv3 = GraphSetConv(
            hidden_dim,
            hidden_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            mhsa_dropout=dropout,
            ffn_dropout=dropout,
        )

        # Attentional graph pool, then attentional set pool.
        self.graph_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )
        )
        self.set_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, set_batch):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.setconv1(x, edge_index, batch, set_batch)
        x = self.setconv2(x, edge_index, batch, set_batch)
        # x = self.setconv3(x, edge_index, batch, set_batch)
        graph_emb = self.graph_pool(x, batch)  # [G, D]
        set_emb = self.set_pool(graph_emb, set_batch)  # [S, D]
        return self.classifier(set_emb)


class GCNGraphClassifier(nn.Module):
    """Simple GCN baseline - classifies graphs individually without set-level aggregation.

    Uses the same GCN encoder architecture as SetTransformer and DeepSets models,
    but predicts directly from graph embeddings. When used with sets, it averages
    the predictions of all graphs in the set.
    """

    def __init__(self, in_channels, hidden_dim, num_classes, dropout=0.1):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, set_batch):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN encoder (same as SetTransformer/DeepSets)
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv3(x, edge_index))
        x = self.dropout(x)

        # Graph-level embedding
        graph_emb = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

        # Get per-graph logits
        graph_logits = self.classifier(graph_emb)  # [num_graphs, num_classes]

        # Average logits within each set to get set-level prediction
        set_logits = scatter_mean(graph_logits, set_batch, dim=0)

        return set_logits


def make_label_homogeneous_sets(dataset, set_size):
    # Group by label
    label_groups = defaultdict(list)
    for data in dataset:
        # Handle both single-label and multi-label datasets
        if data.y.numel() == 1:
            label = int(data.y.item())
        else:
            # For multi-task datasets, use the first task
            label = int(data.y[0].item())
        label_groups[label].append(data)

    sets = []

    for label, graphs in label_groups.items():
        random.shuffle(graphs)
        for i in range(0, len(graphs), set_size):
            sets.append((graphs[i : i + set_size], label))

    random.shuffle(sets)
    return sets
