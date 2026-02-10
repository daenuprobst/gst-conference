import math
import random
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    aggr,
)
from torch_geometric.utils import scatter
from torch_geometric.utils import to_dense_batch


# Wrappers to replace torch_scatter functions
def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


def scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")


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

class GraphSetConv(nn.Module):
    def __init__(
        self,
        filters,
        in_channels=3,
        activation="relu",
        mhsa_dropout=0.1,
        ffn_dropout=0.2,
        pooling="mean",
        use_gating=True,
        ffn_multiplier=8,
        num_heads=4,
    ):
        super().__init__()
        self.filters = filters
        self.activation = activation
        self.pooling = pooling
        self.use_gating = use_gating

        self.gcn_layer = GCNConv(in_channels, filters, improved=True)

        self.gcn_norms = nn.BatchNorm1d(filters)
        self.gcn_dropout = nn.Dropout(ffn_dropout)

        if num_heads == 0:
            num_heads = max(1, filters // 16)

        self.ln_pre = nn.LayerNorm(filters)
        self.mha = nn.MultiheadAttention(
            embed_dim=filters,
            num_heads=num_heads,
            dropout=mhsa_dropout,
            batch_first=True,
        )
        self.ln_post_attn = nn.LayerNorm(filters)

        self.ffn = nn.Sequential(
            nn.Linear(filters, filters * ffn_multiplier),
            self._build_activation(activation),
            nn.Dropout(ffn_dropout),
            nn.Linear(filters * ffn_multiplier, filters),
            nn.Dropout(ffn_dropout),
        )
        self.ln_post_ffn = nn.LayerNorm(filters)

        if use_gating:
            self.gate = nn.Sequential(nn.Linear(filters * 2, filters), nn.Sigmoid())

        self.act = self._build_activation(activation)

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
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return (mean_pool + max_pool) / 2  # Average to keep dimension
        else:
            return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch, set_batch):
        x_input = x

        x = self.gcn_layer(x, edge_index)
        x = self.gcn_norms(x)
        x = self.act(x)
        x = self.gcn_dropout(x)

        z = self._pool_graphs(x, batch)

        z_dense, mask = to_dense_batch(z, set_batch)
        mask = mask.to(dtype=torch.bool, device=z_dense.device)

        z_norm = self.ln_pre(z_dense)

        z_attn, attn_weights = self.mha(z_norm, z_norm, z_norm, key_padding_mask=~mask)
        z_dense = self.ln_post_attn(z_dense + z_attn)

        z_ffn = self.ffn(z_dense)
        z_dense = self.ln_post_ffn(z_dense + z_ffn)

        z_out = z_dense[mask]

        set_info = z_out[batch]

        if self.use_gating:
            gate_input = torch.cat([x, set_info], dim=-1)
            gate_values = self.gate(gate_input)
            x_out = gate_values * set_info + (1 - gate_values) * x
        else:
            x_out = x + set_info

        return x_out


class SetGraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.setconv1 = GraphSetConv(
            filters=hidden_dim, 
            in_channels=in_channels, 
            activation="relu",
            mhsa_dropout=dropout,  
            ffn_dropout=dropout    
        )
        self.setconv2 = GraphSetConv(
            filters=hidden_dim, 
            in_channels=hidden_dim, 
            activation="relu",
            mhsa_dropout=dropout,
            ffn_dropout=dropout
        )
        self.setconv3 = GraphSetConv(
            filters=hidden_dim, 
            in_channels=hidden_dim, 
            activation="relu",
            mhsa_dropout=dropout,
            ffn_dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)  # External dropout between layers (matches other models)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, set_batch):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.setconv1(x, edge_index, batch, set_batch)
        x = self.dropout(x)  # Apply dropout
        x = self.setconv2(x, edge_index, batch, set_batch)
        x = self.dropout(x)  # Apply dropout
        x = self.setconv3(x, edge_index, batch, set_batch)
        x = self.dropout(x)  # Apply dropout
        graph_emb = global_mean_pool(x, batch)
        set_emb = scatter_add(graph_emb, set_batch, dim=0)
        return self.classifier(set_emb)


# Simple GCN baseline (processes individual graphs, no sets)
class GCNGraphClassifier(nn.Module):
    """Simple GCN for single graph classification (no set processing)."""
    
    def __init__(self, in_channels, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data):
        """Process individual graphs (no set_batch needed)."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers with dropout
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv3(x, edge_index))
        x = self.dropout(x)
        
        # Pool to get graph-level embeddings
        graph_emb = global_mean_pool(x, batch)
        
        # Classify
        return self.classifier(graph_emb)


