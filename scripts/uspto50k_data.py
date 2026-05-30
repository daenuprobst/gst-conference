"""
uspto50k_data.py
================

Loader for USPTO-50K (the Schneider et al. classification subset of
Lowe's USPTO grants extraction, with the Dai et al. cleaning and
predefined splits).

Expected on-disk layout
-----------------------
A directory containing three CSVs:
    <data-dir>/raw_train.csv
    <data-dir>/raw_val.csv
    <data-dir>/raw_test.csv

Columns:
    id            patent reaction id
    class         reaction class index (1..10)
    reactants>reagents>production    atom-mapped reaction SMILES,
                                     e.g. "[C:1]...>>...[O:5]"

(Some distributions use `train.csv` / `val.csv` / `test.csv`; pass
`--train-file`, `--val-file`, `--test-file` to override the names.)

Sources
-------
1) Standard GLN split (Dai et al. 2019 cleaning + Schneider 2016
   classification):
        https://github.com/Hanjun-Dai/GLN
        figshare mirror: https://figshare.com/articles/dataset/USPTO-50K_raw_/25459573

2) LocalMapper-remapped variant (Chen et al. 2024) — same column
   layout, cleaner atom mappings (~98.5% accuracy on the chemist-
   labeled subset). Recommended:
        https://figshare.com/articles/dataset/USPTO_reaction_datasets_remapped_by_LocalMapper/25046471

A note on splits
----------------
The benchmark uses these PREDEFINED splits as-is. A random re-shuffle
over USPTO-50K leaks: reactions from the same patent share
intermediates, so a random split inflates val/test scores. Use the
file-defined splits.

A note on the "atom-order leak"
-------------------------------
RetroXpert (Yan et al. 2020) noted that in raw USPTO products the
first SMILES atom is often a reaction center. This is a real concern
for sequence models that consume SMILES tokens, and for any model
that uses atom-position-in-SMILES as a feature.

Our atom-mapping benchmark uses graph neural networks that are
permutation-equivariant in the atom dimension and uses atom map
numbers only as TARGETS, not as features. There is no computational
path through which the leak can reach predictions. We therefore skip
canonical re-ordering by default. The `canonicalize=True` option
applies RetroXpert-style reorder-and-renumber for cross-paper
comparability, but is not required for the architectural comparison
we are running.

Atom features
-------------
Same featurizer as golden_data.py. ATOM_FEAT_DIM = 47.

Dependencies: RDKit, PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError as e:
    raise ImportError(
        "uspto50k_data.py requires RDKit. Install via `pip install rdkit` "
        "or `conda install -c conda-forge rdkit`."
    ) from e


# =============================================================================
# Atom features (identical to golden_data.py)
# =============================================================================

ATOMIC_NUMS: List[int] = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREES: List[int] = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES: List[int] = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS: List = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUM_HS: List[int] = [0, 1, 2, 3, 4]
CHIRALITIES: List = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]


def _one_hot(value, choices) -> List[float]:
    out = [0.0] * (len(choices) + 1)
    try:
        out[choices.index(value)] = 1.0
    except ValueError:
        out[-1] = 1.0
    return out


ATOM_FEAT_DIM: int = (
    (len(ATOMIC_NUMS) + 1)
    + (len(DEGREES) + 1)
    + (len(FORMAL_CHARGES) + 1)
    + (len(HYBRIDIZATIONS) + 1)
    + 1
    + 1
    + (len(NUM_HS) + 1)
    + (len(CHIRALITIES) + 1)
)


def atom_features(atom: Chem.Atom) -> List[float]:
    feats: List[float] = []
    feats += _one_hot(atom.GetAtomicNum(), ATOMIC_NUMS)
    feats += _one_hot(atom.GetDegree(), DEGREES)
    feats += _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))
    feats += _one_hot(atom.GetTotalNumHs(), NUM_HS)
    feats += _one_hot(atom.GetChiralTag(), CHIRALITIES)
    return feats


# =============================================================================
# Graph data classes
# =============================================================================


@dataclass
class MappedMol2DGraph:
    """Single molecule as a 2D graph plus its atom mapping numbers."""

    x: torch.Tensor  # [N, ATOM_FEAT_DIM]
    edge_index: torch.Tensor  # [2, E]   (undirected, both directions)
    num_nodes: int
    atom_map_nums: List[int]  # [N]      mapping number per atom (0 = unmapped)


@dataclass
class AtomMappingExample:
    """One reaction with its ground-truth atom mapping.

    `product_to_reactant[i]` is the flat reactant index that product
    atom i maps to, or -1 if unmapped.
    """

    rxn_id: str
    reactant_mols: List[MappedMol2DGraph]
    product_mols: List[MappedMol2DGraph]
    product_to_reactant: List[int]

    @property
    def num_reactant_atoms(self) -> int:
        return sum(g.num_nodes for g in self.reactant_mols)

    @property
    def num_product_atoms(self) -> int:
        return sum(g.num_nodes for g in self.product_mols)

    @property
    def num_mapped_product_atoms(self) -> int:
        return sum(1 for t in self.product_to_reactant if t >= 0)


# =============================================================================
# SMILES -> graph
# =============================================================================


def _canonical_reorder(mol: Chem.Mol) -> Chem.Mol:
    """RetroXpert-style canonical reordering.

    Strips atom map numbers, computes canonical atom rank without them,
    reorders atoms by that rank, then RESTORES the original map numbers
    on the reordered atoms. This way the canonical order does not
    depend on which atoms are reactive, while the mapping correspondence
    is preserved.
    """
    n = mol.GetNumAtoms()
    if n == 0:
        return mol
    saved = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    except Exception:
        for a, m in zip(mol.GetAtoms(), saved):
            a.SetAtomMapNum(m)
        return mol
    # Restore map numbers on the ORIGINAL mol object first (so the
    # restored saved[i] sticks even after RenumberAtoms below).
    for a, m in zip(mol.GetAtoms(), saved):
        a.SetAtomMapNum(m)
    # ranks[i] = canonical rank of original atom i. We want
    # new_order[k] = original index of the kth canonical atom.
    new_order = sorted(range(n), key=lambda i: ranks[i])
    return Chem.RenumberAtoms(mol, new_order)


def smiles_to_graph(smi: str, canonicalize: bool = False) -> Optional[MappedMol2DGraph]:
    """Parse one (possibly atom-mapped) SMILES into a graph. Returns
    None if parsing fails or the molecule has zero atoms."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if canonicalize:
        mol = _canonical_reorder(mol)
    feats: List[List[float]] = []
    map_nums: List[int] = []
    for atom in mol.GetAtoms():
        feats.append(atom_features(atom))
        map_nums.append(atom.GetAtomMapNum())
    x = torch.tensor(feats, dtype=torch.float32)
    src: List[int] = []
    dst: List[int] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([i, j])
        dst.extend([j, i])
    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return MappedMol2DGraph(
        x=x,
        edge_index=edge_index,
        num_nodes=len(feats),
        atom_map_nums=map_nums,
    )


# =============================================================================
# Mapped RXN-SMILES -> AtomMappingExample
# =============================================================================


def parse_mapped_rxn_smiles(
    rxn_smi: str,
    rxn_id: str = "",
    drop_unmapped_reactant_components: bool = False,
    canonicalize: bool = False,
) -> Optional[AtomMappingExample]:
    """Parse 'reactants>agents>products' (with mapping numbers) into
    an AtomMappingExample. Reactants and agents are merged into the
    reactant side. Returns None on any parse failure or if no product
    atom has a usable mapping."""
    parts = rxn_smi.split(">")
    if len(parts) != 3:
        return None
    reactants_str, agents_str, products_str = parts
    reactant_smis = [s for s in (reactants_str + "." + agents_str).split(".") if s]
    product_smis = [s for s in products_str.split(".") if s]
    if not reactant_smis or not product_smis:
        return None

    reactant_mols: List[MappedMol2DGraph] = []
    for smi in reactant_smis:
        g = smiles_to_graph(smi, canonicalize=canonicalize)
        if g is None:
            return None
        if drop_unmapped_reactant_components and not any(g.atom_map_nums):
            continue
        reactant_mols.append(g)
    product_mols: List[MappedMol2DGraph] = []
    for smi in product_smis:
        g = smiles_to_graph(smi, canonicalize=canonicalize)
        if g is None:
            return None
        product_mols.append(g)
    if not reactant_mols or not product_mols:
        return None

    # reactant map_num -> reactant flat idx (first occurrence wins).
    r_mapnum_to_flat: Dict[int, int] = {}
    flat_idx = 0
    for g in reactant_mols:
        for mn in g.atom_map_nums:
            if mn > 0 and mn not in r_mapnum_to_flat:
                r_mapnum_to_flat[mn] = flat_idx
            flat_idx += 1

    product_to_reactant: List[int] = []
    for g in product_mols:
        for mn in g.atom_map_nums:
            if mn > 0 and mn in r_mapnum_to_flat:
                product_to_reactant.append(r_mapnum_to_flat[mn])
            else:
                product_to_reactant.append(-1)
    if not any(t >= 0 for t in product_to_reactant):
        return None

    return AtomMappingExample(
        rxn_id=rxn_id,
        reactant_mols=reactant_mols,
        product_mols=product_mols,
        product_to_reactant=product_to_reactant,
    )


# =============================================================================
# CSV loader
# =============================================================================

# Default column names from the GLN distribution.
DEFAULT_RXN_COL = "reactants>reagents>production"
DEFAULT_ID_COL = "id"

# Common alternates we'll auto-detect.
ALT_RXN_COLS = (
    "reactants>reagents>production",
    "reactants>reagents>products",
    "rxn_smiles",
    "reaction_smiles",
    "smiles",
)


def _detect_rxn_column(header: List[str]) -> Optional[str]:
    for c in ALT_RXN_COLS:
        if c in header:
            return c
    # Fallback: any column whose name contains '>' (the rxn arrow).
    for c in header:
        if ">" in c:
            return c
    return None


def load_uspto50k_split(
    path: str,
    rxn_col: Optional[str] = None,
    id_col: str = DEFAULT_ID_COL,
    drop_unmapped_reactant_components: bool = False,
    canonicalize: bool = False,
    max_examples: Optional[int] = None,
    max_reactant_atoms: Optional[int] = None,
    max_product_atoms: Optional[int] = None,
    verbose: bool = True,
    label: str = "",
) -> List[AtomMappingExample]:
    """Load one CSV split file. Returns a list of AtomMappingExample.

    Reactions that fail to parse, have no usable mapping, or exceed
    atom-count limits are dropped (counted and reported when verbose).
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path_obj)

    examples: List[AtomMappingExample] = []
    n_total = 0
    n_parse_fail = 0
    n_size_fail = 0

    with open(path_obj) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path_obj}")
        if rxn_col is None:
            rxn_col_eff = _detect_rxn_column(list(reader.fieldnames))
            if rxn_col_eff is None:
                raise ValueError(
                    f"Could not auto-detect a reaction-SMILES column in "
                    f"{path_obj}. Header: {reader.fieldnames}. "
                    f"Pass --rxn-col explicitly."
                )
        else:
            rxn_col_eff = rxn_col
            if rxn_col_eff not in reader.fieldnames:
                raise ValueError(
                    f"Column {rxn_col_eff!r} not in {path_obj}. "
                    f"Header: {reader.fieldnames}"
                )
        id_col_eff = id_col if id_col in reader.fieldnames else None

        for row in reader:
            n_total += 1
            rxn_smi = row.get(rxn_col_eff, "")
            if not rxn_smi:
                n_parse_fail += 1
                continue
            rxn_id = row.get(id_col_eff, "") if id_col_eff else ""
            ex = parse_mapped_rxn_smiles(
                rxn_smi,
                rxn_id=rxn_id,
                drop_unmapped_reactant_components=drop_unmapped_reactant_components,
                canonicalize=canonicalize,
            )
            if ex is None:
                n_parse_fail += 1
                continue
            if (
                max_reactant_atoms is not None
                and ex.num_reactant_atoms > max_reactant_atoms
            ):
                n_size_fail += 1
                continue
            if (
                max_product_atoms is not None
                and ex.num_product_atoms > max_product_atoms
            ):
                n_size_fail += 1
                continue
            examples.append(ex)
            if max_examples is not None and len(examples) >= max_examples:
                break

    if verbose:
        tag = f"[{label}] " if label else ""
        print(
            f"[uspto50k_data] {tag}{path_obj.name}: "
            f"loaded {len(examples):,} / {n_total:,} reactions "
            f"(parse_fail={n_parse_fail}, size_fail={n_size_fail})"
        )
    return examples


def load_uspto50k(
    data_dir: str,
    train_file: str = "raw_train.csv",
    val_file: str = "raw_val.csv",
    test_file: str = "raw_test.csv",
    rxn_col: Optional[str] = None,
    id_col: str = DEFAULT_ID_COL,
    drop_unmapped_reactant_components: bool = False,
    canonicalize: bool = False,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    max_reactant_atoms: Optional[int] = None,
    max_product_atoms: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[
    List[AtomMappingExample], List[AtomMappingExample], List[AtomMappingExample]
]:
    """Load the standard train/val/test predefined splits."""
    d = Path(data_dir)
    if not d.exists():
        raise FileNotFoundError(f"data-dir not found: {d}")

    train = load_uspto50k_split(
        d / train_file,
        rxn_col=rxn_col,
        id_col=id_col,
        drop_unmapped_reactant_components=drop_unmapped_reactant_components,
        canonicalize=canonicalize,
        max_examples=max_train,
        max_reactant_atoms=max_reactant_atoms,
        max_product_atoms=max_product_atoms,
        verbose=verbose,
        label="train",
    )
    val = load_uspto50k_split(
        d / val_file,
        rxn_col=rxn_col,
        id_col=id_col,
        drop_unmapped_reactant_components=drop_unmapped_reactant_components,
        canonicalize=canonicalize,
        max_examples=max_val,
        max_reactant_atoms=max_reactant_atoms,
        max_product_atoms=max_product_atoms,
        verbose=verbose,
        label="val",
    )
    test = load_uspto50k_split(
        d / test_file,
        rxn_col=rxn_col,
        id_col=id_col,
        drop_unmapped_reactant_components=drop_unmapped_reactant_components,
        canonicalize=canonicalize,
        max_examples=max_test,
        max_reactant_atoms=max_reactant_atoms,
        max_product_atoms=max_product_atoms,
        verbose=verbose,
        label="test",
    )
    return train, val, test


# =============================================================================
# CLI: smoke-test the loader on one file
# =============================================================================


def _cli_inspect(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Inspect a USPTO-50K CSV split file (smoke test).",
    )
    p.add_argument("--csv", required=True, help="path to a raw_*.csv file")
    p.add_argument("--rxn-col", default=None)
    p.add_argument("--id-col", default=DEFAULT_ID_COL)
    p.add_argument("--canonicalize", action="store_true")
    p.add_argument("--max-examples", type=int, default=None)
    args = p.parse_args(argv)

    examples = load_uspto50k_split(
        args.csv,
        rxn_col=args.rxn_col,
        id_col=args.id_col,
        canonicalize=args.canonicalize,
        max_examples=args.max_examples,
    )
    if not examples:
        return 1
    n_r = sum(ex.num_reactant_atoms for ex in examples)
    n_p = sum(ex.num_product_atoms for ex in examples)
    n_m = sum(ex.num_mapped_product_atoms for ex in examples)
    print(
        f"  reactions               : {len(examples):,}\n"
        f"  reactant atoms (total)  : {n_r:,}\n"
        f"  product  atoms (total)  : {n_p:,}\n"
        f"  mapped product atoms    : {n_m:,}  "
        f"({100.0 * n_m / max(n_p, 1):.1f}%)"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] not in {"inspect"}:
        print(
            "Usage:\n  python uspto50k_data.py inspect --csv <path>\n",
            file=sys.stderr,
        )
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "inspect":
        return _cli_inspect(rest)
    return 2


if __name__ == "__main__":
    sys.exit(main())
