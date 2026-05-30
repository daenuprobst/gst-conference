"""
Timing benchmark: GST-bc / GST-ca vs GCN+DeepSets / GCN+SetTransformer.
Runs as-is. Prints mean step time (forward + loss + backward + optimizer).
3 layers, hidden=128, synthetic ER graph sets (4 graphs/set, 32 sets/batch).

Uses the real GraphSetConv block from graph_set_transformer.models.gst_v2.
Make sure your project is importable (run from the repo root, or add it
to PYTHONPATH).

Deps: torch, torch_geometric, numpy.
"""

import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

from graph_set_transformer.models.gst import GraphSetConv

# ---- Config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = 3
HIDDEN = 128
NUM_HEADS = 4
BATCH_SIZE = 32
SET_SIZE = 4
N_RANGE = (12, 25)
P_RANGE = (0.10, 0.25)
FEAT_DIM = 64
NUM_BATCHES = 8
WARMUP_STEPS = 10
TIMED_STEPS = 50
SEED = 0


# ---- Pooling ----
def mean_pool(x, batch, n):
    out = x.new_zeros(n, x.size(-1))
    out.index_add_(0, batch, x)
    cnt = x.new_zeros(n)
    cnt.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return out / cnt.clamp_min(1.0).unsqueeze(-1)


def add_pool(x, batch, n):
    out = x.new_zeros(n, x.size(-1))
    out.index_add_(0, batch, x)
    return out


# ---- Pipeline baselines ----
class GCNStack(nn.Module):
    def __init__(self, h, L):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(h, h, improved=True) for _ in range(L)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])

    def forward(self, x, ei):
        for c, n in zip(self.convs, self.norms):
            x = F.silu(n(c(x, ei)))
        return x


class DeepSets(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))

    def forward(self, zg, sb, ns):
        return self.rho(add_pool(self.phi(zg), sb, ns))


class SetTransformer(nn.Module):
    def __init__(self, h, nh):
        super().__init__()
        self.ln1 = nn.LayerNorm(h)
        self.mha = nn.MultiheadAttention(h, nh, batch_first=True)
        self.ln2 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, h * 2), nn.SiLU(), nn.Linear(h * 2, h))
        self.seed = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.ln3 = nn.LayerNorm(h)
        self.pma = nn.MultiheadAttention(h, nh, batch_first=True)

    def forward(self, zg, sb, ns):
        zd, mask = to_dense_batch(zg, sb, batch_size=ns)
        zn = self.ln1(zd)
        a, _ = self.mha(zn, zn, zn, key_padding_mask=~mask, need_weights=False)
        zd = zd + a
        zd = zd + self.ffn(self.ln2(zd))
        seed = self.seed.expand(zd.size(0), -1, -1)
        z, _ = self.pma(
            seed, self.ln3(zd), self.ln3(zd), key_padding_mask=~mask, need_weights=False
        )
        return z.squeeze(1)


class Pipeline(nn.Module):
    def __init__(self, in_dim, h, L, nh, head):
        super().__init__()
        self.proj = nn.Linear(in_dim, h)
        self.enc = GCNStack(h, L)
        self.head = DeepSets(h) if head == "ds" else SetTransformer(h, nh)
        self.out = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x, ei, b, sb, ng, ns):
        h = self.enc(self.proj(x), ei)
        zg = mean_pool(h, b, ng)
        zs = self.head(zg, sb, ns)
        return self.out(zs).squeeze(-1)


# ---- GraphSetConv wrapper using the real block ----
class GST(nn.Module):
    def __init__(self, in_dim, h, L, nh, mode):
        super().__init__()
        assert mode in ("broadcast", "cross_attn")
        self.proj = nn.Linear(in_dim, h)
        self.blocks = nn.ModuleList(
            [
                GraphSetConv(
                    filters=h,
                    in_channels=h,
                    num_heads=nh,
                    mhsa_dropout=0.0,
                    ffn_dropout=0.0,
                    node_set_mode=mode,
                )
                for _ in range(L)
            ]
        )
        self.out = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x, ei, b, sb, ng, ns):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x, ei, b, sb)
        zg = mean_pool(x, b, ng)
        zs = mean_pool(zg, sb, ns)
        return self.out(zs).squeeze(-1)


MODELS = {
    "GST-bc": lambda: GST(FEAT_DIM, HIDDEN, LAYERS, NUM_HEADS, "broadcast"),
    "GST-ca": lambda: GST(FEAT_DIM, HIDDEN, LAYERS, NUM_HEADS, "cross_attn"),
    "GCN+DeepSets": lambda: Pipeline(FEAT_DIM, HIDDEN, LAYERS, NUM_HEADS, "ds"),
    "GCN+SetTransformer": lambda: Pipeline(FEAT_DIM, HIDDEN, LAYERS, NUM_HEADS, "st"),
}


# ---- Synthetic data ----
def make_batch(seed):
    rng = np.random.default_rng(seed)
    xs, eis, bs, ss = [], [], [], []
    n_off, g_off = 0, 0
    for sid in range(BATCH_SIZE):
        for _ in range(SET_SIZE):
            n = int(rng.integers(N_RANGE[0], N_RANGE[1] + 1))
            p = float(rng.uniform(*P_RANGE))
            adj = np.triu(rng.random((n, n)) < p, k=1)
            src, dst = np.where(adj)
            if src.size:
                e = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
                ei = torch.from_numpy(e).long() + n_off
            else:
                ei = torch.zeros((2, 0), dtype=torch.long)
            xs.append(
                torch.from_numpy(rng.standard_normal((n, FEAT_DIM)).astype(np.float32))
            )
            eis.append(ei)
            bs.append(torch.full((n,), g_off, dtype=torch.long))
            n_off += n
            g_off += 1
        ss.append(torch.full((SET_SIZE,), sid, dtype=torch.long))
    return {
        "x": torch.cat(xs).to(DEVICE),
        "ei": (
            torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long)
        ).to(DEVICE),
        "b": torch.cat(bs).to(DEVICE),
        "sb": torch.cat(ss).to(DEVICE),
        "y": torch.zeros(BATCH_SIZE).to(DEVICE),
        "ng": g_off,
        "ns": BATCH_SIZE,
    }


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def time_model(name, builder, batches):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = builder().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for i in range(WARMUP_STEPS):
        b = batches[i % len(batches)]
        opt.zero_grad(set_to_none=True)
        out = model(b["x"], b["ei"], b["b"], b["sb"], b["ng"], b["ns"])
        F.mse_loss(out, b["y"]).backward()
        opt.step()
    sync()

    times_ms = []
    for i in range(TIMED_STEPS):
        b = batches[i % len(batches)]
        sync()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(b["x"], b["ei"], b["b"], b["sb"], b["ng"], b["ns"])
        F.mse_loss(out, b["y"]).backward()
        opt.step()
        sync()
        times_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "mean": float(np.mean(times_ms)),
        "std": float(np.std(times_ms, ddof=1)),
        "median": float(statistics.median(times_ms)),
        "params": n_params,
    }


def main():
    print(
        f"device={DEVICE}, depth={LAYERS}, hidden={HIDDEN}, "
        f"batch={BATCH_SIZE} sets x {SET_SIZE} graphs, "
        f"warmup={WARMUP_STEPS}, timed={TIMED_STEPS}"
    )
    print()
    batches = [make_batch(SEED + i) for i in range(NUM_BATCHES)]
    results = {name: time_model(name, b, batches) for name, b in MODELS.items()}

    base = min(r["mean"] for r in results.values())
    print(
        f"{'Model':<22} {'ms/step (mean ± std)':<24} {'median':>10} "
        f"{'rel':>6} {'params':>12}"
    )
    print("-" * 78)
    for name, r in results.items():
        print(
            f"{name:<22} {r['mean']:>7.2f} ± {r['std']:>5.2f}{'':<8} "
            f"{r['median']:>10.2f} {r['mean'] / base:>5.2f}x {r['params']:>12,}"
        )


if __name__ == "__main__":
    main()
