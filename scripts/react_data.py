"""
bh_data.py
==========

Loader for the Buchwald-Hartwig HTE yield-prediction dataset (Ahneman et al.,
Science 2018; Sandfort et al., Chem 2020 splits hosted by Schwaller's
rxn_yields repo).

Each row of the xlsx represents one HTE reaction. The dataset is a complete
factorial of:
    15 aryl halides × 4 ligands × 3 bases × 23 isoxazole additives = 4140
plus controls = 4608 rows total. Yield (Output column) ranges 0-100%.

Splits available (sheet names in the xlsx):
    FullCV_01..FullCV_10   - 10 random 70/30 splits (paper headline numbers)
    Test1..Test4           - leave-out-additive splits (extrapolation; harder)

For our benchmark, each reaction is a SET of 4 small molecular graphs:
{aryl_halide, ligand, base, additive}. We deliberately skip the methylaniline
(constant across all 4140 rows) and Pd catalyst (also constant) since they
carry zero information signal.

User decision (from this conversation): a single shared encoder is used across
all 4 reactant types — the architectural claim is that interleaved set-graph
reasoning helps, not that per-slot specialization helps.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Source URL for the canonical xlsx. We download once and cache.
DATA_URL = (
    "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/"
    "data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
)
DATA_FILENAME = "Dreher_and_Doyle_input_data.xlsx"

REACTANT_COLUMNS = ["Aryl halide", "Ligand", "Base", "Additive"]
TARGET_COLUMN = "Output"

# Atom features — choice rationale below. We support two modes via the
# `feature_mode` argument to load_bh_full / load_bh_split:
#
#   "minimal":  atomic-number one-hot + aromatic flag (12 dims). Original
#               behavior, kept for backward compatibility. Fastest, most
#               architecture-stress-testing.
#
#   "rich":    atomic-number one-hot + aromatic + formal-charge + hybridization
#               + in-ring + degree + num-Hs (~30 dims). Standard chemistry-GNN
#               feature set used in AGNN, ChemProp, MPNN papers. Recovers
#               typical yield-prediction performance closer to literature
#               numbers, which depend heavily on these descriptors.
ALLOWED_ATOMS = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)  # H C N O F P S Cl Br I

# Hybridization buckets (RDKit values we'll one-hot over)
_HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER")
# Degree buckets (truncated; >5 lumped into the last)
_DEGREE_BUCKETS = (0, 1, 2, 3, 4, 5)
# Num-Hs buckets
_NUMH_BUCKETS = (0, 1, 2, 3, 4)
# Formal-charge buckets (-2, -1, 0, +1, +2; outliers lumped to 0)
_CHARGE_BUCKETS = (-2, -1, 0, 1, 2)


def _atom_feat_dim(mode: str) -> int:
    if mode == "minimal":
        return len(ALLOWED_ATOMS) + 1 + 1
    if mode == "rich":
        return (
            len(ALLOWED_ATOMS)
            + 1  # element one-hot + "other"
            + 1  # aromatic flag
            + len(_CHARGE_BUCKETS)
            + 1  # formal charge one-hot + "other"
            + len(_HYBRIDIZATIONS)  # hybridization one-hot
            + 1  # in-ring flag
            + len(_DEGREE_BUCKETS)  # degree one-hot
            + len(_NUMH_BUCKETS)  # num-Hs one-hot
        )
    raise ValueError(f"Unknown feature mode: {mode!r}. Valid: 'minimal', 'rich'.")


# Legacy module-level constant. Default is 'minimal' so older code paths
# that didn't pass `feature_mode` continue to work bit-for-bit; the BH
# benchmark threads through the chosen mode explicitly.
ATOM_FEAT_DIM = _atom_feat_dim("minimal")


def get_atom_feat_dim(feature_mode: str) -> int:
    """Public helper for callers (benchmark script) to size their input
    projection layer correctly."""
    return _atom_feat_dim(feature_mode)


def _ensure_rdkit():
    try:
        from rdkit import Chem

        return Chem
    except ImportError as e:
        raise ImportError(
            "RDKit is required. Install with `pip install rdkit` or "
            "`conda install -c conda-forge rdkit`."
        ) from e


def _atom_features_minimal(atom) -> torch.Tensor:
    """Atomic-number one-hot + aromatic flag (12 dims)."""
    Z = atom.GetAtomicNum()
    feats = [float(Z == z) for z in ALLOWED_ATOMS]
    feats.append(0.0 if Z in ALLOWED_ATOMS else 1.0)  # "other" bucket
    feats.append(float(atom.GetIsAromatic()))
    return torch.tensor(feats, dtype=torch.float32)


def _atom_features_rich(atom) -> torch.Tensor:
    """Standard chemistry-GNN feature set: element + aromatic + charge +
    hybridization + ring + degree + num-Hs. Returns [~30] float."""
    Z = atom.GetAtomicNum()
    feats: List[float] = []

    # element one-hot + "other"
    feats.extend(float(Z == z) for z in ALLOWED_ATOMS)
    feats.append(0.0 if Z in ALLOWED_ATOMS else 1.0)

    # aromatic
    feats.append(float(atom.GetIsAromatic()))

    # formal charge one-hot + "other"
    fc = atom.GetFormalCharge()
    feats.extend(float(fc == c) for c in _CHARGE_BUCKETS)
    feats.append(0.0 if fc in _CHARGE_BUCKETS else 1.0)

    # hybridization one-hot
    hyb = str(atom.GetHybridization()).split(".")[
        -1
    ]  # 'SP3' from 'HybridizationType.SP3'
    feats.extend(float(hyb == h) for h in _HYBRIDIZATIONS[:-1])
    # last bucket is OTHER (everything not in _HYBRIDIZATIONS[:-1])
    feats.append(1.0 if hyb not in _HYBRIDIZATIONS[:-1] else 0.0)

    # in-ring
    feats.append(float(atom.IsInRing()))

    # degree one-hot (>5 lumped to 5)
    deg = min(atom.GetDegree(), _DEGREE_BUCKETS[-1])
    feats.extend(float(deg == d) for d in _DEGREE_BUCKETS)

    # num-Hs one-hot (>4 lumped to 4)
    nh = min(atom.GetTotalNumHs(), _NUMH_BUCKETS[-1])
    feats.extend(float(nh == n) for n in _NUMH_BUCKETS)

    return torch.tensor(feats, dtype=torch.float32)


def _atom_features(atom, feature_mode: str = "minimal") -> torch.Tensor:
    if feature_mode == "minimal":
        return _atom_features_minimal(atom)
    if feature_mode == "rich":
        return _atom_features_rich(atom)
    raise ValueError(f"Unknown feature mode: {feature_mode!r}")


@dataclass
class Mol2DGraph:
    """One 2D molecular graph: node features and bond edges."""

    x: torch.Tensor  # [n, ATOM_FEAT_DIM]
    edge_index: torch.Tensor  # [2, E] (directed; both directions stored)

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]


@dataclass
class BHReaction:
    """One Buchwald-Hartwig reaction: 4 reactant graphs + scalar yield target.
    The set of 4 graphs is the input the architecture sees.

    `component_smiles` (added later for the stratified-by-additive val carve)
    holds the canonical SMILES of each reactant in the same order as
    `graphs`: [aryl_halide, ligand, base, additive]. Old caches that don't
    have this field are accepted; component_smiles defaults to an empty
    list and stratified splits will fall back to random.
    """

    name: str
    graphs: List[Mol2DGraph]
    y: float
    component_smiles: List[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.component_smiles is None:
            self.component_smiles = []

    @property
    def additive_smiles(self) -> Optional[str]:
        """The additive is the 4th component (index 3)."""
        if len(self.component_smiles) >= 4:
            return self.component_smiles[3]
        return None


def _mol_from_smiles(smiles: str):
    Chem = _ensure_rdkit()
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def _mol_to_graph(mol, feature_mode: str = "minimal") -> Optional[Mol2DGraph]:
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    n = mol.GetNumAtoms()
    x = torch.stack(
        [
            _atom_features(mol.GetAtomWithIdx(i), feature_mode=feature_mode)
            for i in range(n)
        ],
        dim=0,
    )
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Mol2DGraph(x=x, edge_index=edge_index)


# =============================================================================
# Download + sheet readers
# =============================================================================


def ensure_data_downloaded(data_root: Path) -> Path:
    """Download the xlsx into data_root if missing. Returns the file path."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    xlsx_path = data_root / DATA_FILENAME
    if xlsx_path.exists():
        return xlsx_path
    print(f"  [bh] downloading {DATA_URL} -> {xlsx_path}")
    import urllib.request

    urllib.request.urlretrieve(DATA_URL, xlsx_path)
    return xlsx_path


def _read_sheet(xlsx_path: Path, sheet_name: str):
    import pandas as pd

    df = pd.read_excel(str(xlsx_path), sheet_name=sheet_name)
    missing = [c for c in REACTANT_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Sheet {sheet_name!r} is missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def _row_to_reaction(
    row, name: str, graph_cache: Dict[str, Mol2DGraph], feature_mode: str = "minimal"
) -> Optional[BHReaction]:
    """Convert a dataframe row to a BHReaction, with SMILES->graph caching
    (every distinct reactant SMILES appears many times in the factorial,
    so caching the parsed graphs gives a ~50× speedup on parsing).

    The graph_cache is keyed by SMILES alone; the caller should pass a
    fresh cache when feature_mode changes."""
    graphs = []
    component_smiles: List[str] = []
    for col in REACTANT_COLUMNS:
        smi = row.get(col, None)
        if smi is None or (isinstance(smi, float) and math.isnan(smi)):
            return None
        smi = str(smi).strip()
        if not smi:
            return None
        component_smiles.append(smi)
        if smi in graph_cache:
            graphs.append(graph_cache[smi])
        else:
            g = _mol_to_graph(_mol_from_smiles(smi), feature_mode=feature_mode)
            if g is None:
                return None
            graph_cache[smi] = g
            graphs.append(g)
    try:
        y_val = float(row[TARGET_COLUMN])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(y_val):
        return None
    return BHReaction(
        name=name, graphs=graphs, y=y_val, component_smiles=component_smiles
    )


# =============================================================================
# Public loaders
# =============================================================================


# Map our --split argument values to xlsx sheet names. The xlsx sheets are:
#   FullCV_01..FullCV_10  -> 10 random 70/30 splits (3955 train / 753 test)
#   Test1..Test4          -> 4 leave-out-additive splits (paper Test sets)
SHEET_NAMES_FULLCV = [f"FullCV_{i:02d}" for i in range(1, 11)]
SHEET_NAMES_TEST = [f"Test{i}" for i in range(1, 5)]


def load_bh_full(
    data_root: Path, feature_mode: str = "minimal", cache_filename: Optional[str] = None
) -> List[BHReaction]:
    """Load the entire dataset (FullCV_01 rows; the 10 FullCV sheets are
    just shuffles of the same factorial). Used for our own random splits.

    feature_mode: 'minimal' (default, 12-d) or 'rich' (~30-d). Caches are
    stamped with the mode so changing it forces a re-parse.
    """
    data_root = Path(data_root)
    if cache_filename is None:
        cache_filename = f"bh_full_cache_{feature_mode}.pt"
    cache_path = data_root / cache_filename
    if cache_path.exists():
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            reactions = cache.get("reactions", [])
            cache_mode = cache.get("feature_mode", "minimal")
            if reactions and cache_mode == feature_mode:
                print(
                    f"  [bh] loaded {len(reactions)} reactions from "
                    f"cache {cache_path.name} (feature_mode={feature_mode})"
                )
                return reactions
            elif reactions and cache_mode != feature_mode:
                print(
                    f"  [bh] cache feature_mode mismatch "
                    f"({cache_mode!r} vs requested {feature_mode!r}); "
                    f"re-parsing."
                )
        except Exception:
            pass

    xlsx_path = ensure_data_downloaded(data_root)
    df = _read_sheet(xlsx_path, "FullCV_01")
    print(
        f"  [bh] parsing {len(df)} rows from FullCV_01 (feature_mode={feature_mode})..."
    )
    graph_cache: Dict[str, Mol2DGraph] = {}
    reactions: List[BHReaction] = []
    n_skip = 0
    for i, row in df.iterrows():
        rxn = _row_to_reaction(
            row, name=f"row_{i:04d}", graph_cache=graph_cache, feature_mode=feature_mode
        )
        if rxn is None:
            n_skip += 1
            continue
        reactions.append(rxn)
    if n_skip:
        print(f"  [bh] skipped {n_skip} rows (parse failures)")
    print(
        f"  [bh] {len(reactions)} reactions retained, "
        f"{len(graph_cache)} unique reactant graphs"
    )

    torch.save({"reactions": reactions, "feature_mode": feature_mode}, cache_path)
    print(f"  [bh] cached to {cache_path.name}")
    return reactions


def load_bh_split(
    data_root: Path,
    split_name: str,
    feature_mode: str = "minimal",
    cache_filename: Optional[str] = None,
) -> Tuple[List[BHReaction], List[BHReaction]]:
    """Load one of the paper's pre-defined splits.

    For FullCV_NN sheets, the convention from rxn_yields is:
        train = rows[:2768] (ish), test = rows[2768:]
    The exact split point is sheet-dependent; we read the actual count from
    the rxn_yields convention (NAME_SPLIT in their notebook):
        FullCV_01..FullCV_10: train = first 2768 rows, test = remainder
        Test1..Test4:         train = first <split> rows (varies), test = rest
    We hardcode these counts since they're a fixed protocol artifact.

    feature_mode: 'minimal' or 'rich'. Caches are stamped with the mode.
    """
    data_root = Path(data_root)
    if cache_filename is None:
        cache_filename = f"bh_{split_name}_cache_{feature_mode}.pt"
    cache_path = data_root / cache_filename
    if cache_path.exists():
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            train, test = cache.get("train", []), cache.get("test", [])
            cache_mode = cache.get("feature_mode", "minimal")
            if train and test and cache_mode == feature_mode:
                print(
                    f"  [bh] loaded {split_name}: {len(train)} train / "
                    f"{len(test)} test from cache "
                    f"(feature_mode={feature_mode})"
                )
                return train, test
            elif train and test and cache_mode != feature_mode:
                print(f"  [bh] cache feature_mode mismatch; re-parsing.")
        except Exception:
            pass

    SPLIT_POINTS = {
        **{f"FullCV_{i:02d}": 2768 for i in range(1, 11)},
        "Test1": 3058,
        "Test2": 3056,
        "Test3": 3059,
        "Test4": 3056,
    }
    if split_name not in SPLIT_POINTS:
        raise ValueError(
            f"Unknown split {split_name!r}. Valid: {list(SPLIT_POINTS.keys())}"
        )
    split_point = SPLIT_POINTS[split_name]

    xlsx_path = ensure_data_downloaded(data_root)
    df = _read_sheet(xlsx_path, split_name)
    print(
        f"  [bh] parsing {split_name}: {len(df)} rows, "
        f"split at {split_point} (feature_mode={feature_mode})..."
    )
    graph_cache: Dict[str, Mol2DGraph] = {}

    train: List[BHReaction] = []
    test: List[BHReaction] = []
    n_skip = 0
    for i, row in df.iterrows():
        rxn = _row_to_reaction(
            row,
            name=f"{split_name}:row_{i:04d}",
            graph_cache=graph_cache,
            feature_mode=feature_mode,
        )
        if rxn is None:
            n_skip += 1
            continue
        # Off-by-one matches rxn_yields exactly: test starts at split-1
        if i < split_point - 1:
            train.append(rxn)
        else:
            test.append(rxn)
    if n_skip:
        print(f"  [bh] skipped {n_skip} rows")
    print(f"  [bh] {split_name}: {len(train)} train / {len(test)} test")

    torch.save({"train": train, "test": test, "feature_mode": feature_mode}, cache_path)
    return train, test


# =============================================================================
# Random 70/10/20 split for our headline runs (sampled within FullCV_01)
# =============================================================================


def random_split(
    items: List[BHReaction],
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[List[BHReaction], List[BHReaction], List[BHReaction]]:
    """Random 70/10/20 split (matches MARCEL/synth conventions)."""
    n = len(items)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    return (
        [items[i] for i in perm[:n_train]],
        [items[i] for i in perm[n_train : n_train + n_val]],
        [items[i] for i in perm[n_train + n_val :]],
    )


def val_split_from_train(
    train: List[BHReaction], seed: int = 0, val_fraction: float = 0.1
) -> Tuple[List[BHReaction], List[BHReaction]]:
    """RANDOM val carve out of train. Use for in-distribution evaluation
    (FullCV_NN). Bad choice for OOD splits (Test1-4) because val sees the
    same component distribution as train, so early stopping picks the
    point that generalizes well to seen components, not unseen ones."""
    n = len(train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return (
        [train[i] for i in train_idx],
        [train[i] for i in val_idx],
    )


def val_split_stratified_by_additive(
    train: List[BHReaction],
    seed: int = 0,
    n_val_additives: Optional[int] = None,
    val_fraction: float = 0.1,
) -> Tuple[List[BHReaction], List[BHReaction]]:
    """STRATIFIED-BY-ADDITIVE val carve. Holds out one or more entire
    additives (rather than random rows) to mimic the OOD test setup.

    Use this for Test1-4 evaluation, where the test set holds out unseen
    additives entirely. Random val undermines early stopping on those
    splits because val rewards in-distribution generalization.

    n_val_additives: how many distinct additives to hold out. If None,
        chosen so the val set is approximately `val_fraction` of train
        rows (rounded up to whole additives, with a floor of 1).

    Falls back to random val if `component_smiles` is missing (e.g. old
    cache without per-component metadata).
    """
    # Are component SMILES populated?
    has_meta = all(len(r.component_smiles) >= 4 for r in train)
    if not has_meta:
        print(
            "  [bh] WARNING: component_smiles missing on train items "
            "(probably from an older cache). Falling back to random "
            "val carve. Delete bh_*_cache_*.pt and re-run to enable "
            "stratified val."
        )
        return val_split_from_train(train, seed=seed, val_fraction=val_fraction)

    # Group rows by additive
    by_additive: Dict[str, List[int]] = {}
    for i, r in enumerate(train):
        add = r.additive_smiles or ""
        by_additive.setdefault(add, []).append(i)

    additives = sorted(by_additive.keys())
    rng = np.random.default_rng(seed)
    perm_add = list(rng.permutation(len(additives)))
    additives_shuffled = [additives[i] for i in perm_add]

    if n_val_additives is None:
        # Pick enough additives so total val rows ~= val_fraction * n_train
        target_rows = max(1, int(round(val_fraction * len(train))))
        chosen: List[str] = []
        rows_so_far = 0
        for add in additives_shuffled:
            if rows_so_far >= target_rows and chosen:
                break
            chosen.append(add)
            rows_so_far += len(by_additive[add])
        if not chosen:
            chosen = [additives_shuffled[0]]
    else:
        chosen = additives_shuffled[: max(1, int(n_val_additives))]

    chosen_set = set(chosen)
    train_out: List[BHReaction] = []
    val_out: List[BHReaction] = []
    for r in train:
        add = r.additive_smiles or ""
        if add in chosen_set:
            val_out.append(r)
        else:
            train_out.append(r)
    print(
        f"  [bh] stratified val carve: held out "
        f"{len(chosen)}/{len(additives)} additives -> "
        f"{len(val_out)} val rows / {len(train_out)} train rows"
    )
    return train_out, val_out


# =============================================================================
# Target normalization
# =============================================================================


@dataclass
class TargetStats:
    mean: torch.Tensor  # scalar
    std: torch.Tensor  # scalar

    def to(self, device) -> "TargetStats":
        return TargetStats(self.mean.to(device), self.std.to(device))

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std.clamp_min(1e-9)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


def compute_target_stats(items: List[BHReaction]) -> TargetStats:
    if not items:
        raise RuntimeError(
            "compute_target_stats received zero items. "
            "Loader returned an empty list — check parse warnings above."
        )
    Y = torch.tensor([it.y for it in items], dtype=torch.float32)
    return TargetStats(mean=Y.mean(), std=Y.std())
