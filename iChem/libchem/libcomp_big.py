from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any
import pickle as pkl
import re

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from ..utils import load_multiple_smiles
from ..visualization.mol_images import smiles_to_grid_image
from ..visualization.plots import venn_lib_comp


def load_mixed_cluster_results(final_cluster_mixed_libs_path: str | Path):
    """Load final mixed-cluster output produced by clustering scripts.

    Parameters:
        final_cluster_mixed_libs_path: Path to final_cluster_mixed_libs.pkl.

    Returns:
        Tuple (final_bfs, final_mol_ids).
    """
    with open(final_cluster_mixed_libs_path, "rb") as f:
        final_bfs, final_mol_ids = pkl.load(f)
    return final_bfs, final_mol_ids


def _normalize_mol_id(mol_id: Any) -> str:
    """Convert molecule ids from strings/bytes/numpy scalar into plain str."""
    if isinstance(mol_id, bytes):
        return mol_id.decode("utf-8", errors="ignore")
    if isinstance(mol_id, np.bytes_):
        return bytes(mol_id).decode("utf-8", errors="ignore")
    if isinstance(mol_id, memoryview):
        return mol_id.tobytes().decode("utf-8", errors="ignore")
    if isinstance(mol_id, np.generic):
        return str(mol_id.item())
    return str(mol_id)


def _extract_lib_label(mol_id: str) -> str:
    """Extract library label prefix from molecule id.

    Expected ids are typically like A123, B45, etc. This function also supports
    multi-letter prefixes by taking the leading alphabetic run.
    """
    match = re.match(r"^[A-Za-z]+", mol_id)
    if match:
        return match.group(0)
    return mol_id[0] if mol_id else "?"


def _extract_local_index(mol_id: str) -> int:
    """Extract numeric index from molecule id.

    Raises:
        ValueError if no numeric part is found.
    """
    match = re.search(r"(\d+)$", mol_id)
    if not match:
        raise ValueError(f"Could not parse numeric index from molecule id: {mol_id}")
    return int(match.group(1))


def extract_mols_from_buffers(final_mol_ids):
    """Flatten buffer-grouped final_mol_ids into a list of clusters.

    The clustering outputs may come as a dict keyed by dtype/buffer groups.
    This function preserves cluster order inside each sorted key.
    """
    mol_ids = []
    for key in sorted(final_mol_ids.keys()):
        for mol_ids_cluster in final_mol_ids[key]:
            normalized_cluster = [_normalize_mol_id(mol_id) for mol_id in mol_ids_cluster]
            mol_ids.append(normalized_cluster)
    return mol_ids


def count_and_map_clusters(final_mol_ids):
    """Count library membership per cluster and build overlap flags.

    Parameters:
        final_mol_ids: Either flat list of clusters or buffer-grouped dict.

    Returns:
        cluster_counts: list[Counter], each Counter maps lib label -> molecule count.
        cluster_flag: list[str], e.g. "A", "B", "A+B".
    """
    cluster_counts = []
    cluster_flag = []

    if isinstance(final_mol_ids, dict):
        final_mol_ids = extract_mols_from_buffers(final_mol_ids)

    for cluster in final_mol_ids:
        labels = [_extract_lib_label(_normalize_mol_id(mol_id)) for mol_id in cluster]
        count = Counter(labels)
        cluster_counts.append(count)
        flag = "+".join([key for key in sorted(count.keys()) if count[key] > 0])
        cluster_flag.append(flag)

    return cluster_counts, cluster_flag


def cluster_indices_by_label(final_mol_ids=None, cluster_flags: list[str] | None = None) -> dict[str, list[int]]:
    """Return cluster indices grouped by flag label.

    Examples of labels: "A", "B", "A+B", "A+B+C".
    """
    if cluster_flags is None:
        _, cluster_flags = count_and_map_clusters(final_mol_ids)

    out: dict[str, list[int]] = {}
    for idx, label in enumerate(cluster_flags):
        out.setdefault(label, []).append(idx)
    return out


def cluster_indices_for_label(
    label: str,
    final_mol_ids=None,
    cluster_flags: list[str] | None = None,
) -> list[int]:
    """Return all cluster indices matching a specific label."""
    by_label = cluster_indices_by_label(final_mol_ids=final_mol_ids, cluster_flags=cluster_flags)
    return by_label.get(label, [])


def cluster_flag_counts(final_mol_ids=None, cluster_flags=None) -> Counter:
    """Return counts of cluster overlap flags.

    Pass either final_mol_ids or precomputed cluster_flags.
    """
    if cluster_flags is None:
        _, cluster_flags = count_and_map_clusters(final_mol_ids)
    return Counter(cluster_flags)


def compute_library_space_totals(cluster_counts: list[Counter]):
    """Compute molecule totals split by exclusive/shared contributions."""
    all_libs = sorted({lib for count in cluster_counts for lib in count.keys()})
    lib_totals = {lib: 0 for lib in all_libs}
    exclusive_totals = {lib: 0 for lib in all_libs}
    shared_totals = {lib: 0 for lib in all_libs}

    for cluster in cluster_counts:
        if len(cluster) == 1:
            lib = next(iter(cluster))
            exclusive_totals[lib] += cluster[lib]
            lib_totals[lib] += cluster[lib]
        else:
            for lib, n in cluster.items():
                shared_totals[lib] += n
                lib_totals[lib] += n

    return lib_totals, exclusive_totals, shared_totals


def compute_cluster_presence_metrics(cluster_counts: list[Counter], cluster_flags: list[str]):
    """Compute cluster-level diversity/shared-space metrics.

    Returns a dict with per-library metrics and overlap percentages.
    """
    all_libs = sorted({lib for flag in cluster_flags for lib in flag.split('+') if lib})
    total_clusters = len(cluster_counts)
    flag_counts = Counter(cluster_flags)

    per_lib = {}
    for lib in all_libs:
        n_clusters_with_lib = sum(1 for count in cluster_counts if lib in count)
        n_clusters_exclusive = flag_counts.get(lib, 0)
        n_clusters_shared = n_clusters_with_lib - n_clusters_exclusive
        per_lib[lib] = {
            "n_clusters_with_lib": n_clusters_with_lib,
            "n_clusters_exclusive": n_clusters_exclusive,
            "n_clusters_shared": n_clusters_shared,
            "pct_clusters_with_lib": 100.0 * n_clusters_with_lib / total_clusters if total_clusters else 0.0,
        }

    pairwise = {}
    for lib_a, lib_b in combinations(all_libs, 2):
        key_ab = "+".join(sorted([lib_a, lib_b]))
        n_ab = flag_counts.get(key_ab, 0)
        n_a = flag_counts.get(lib_a, 0)
        n_b = flag_counts.get(lib_b, 0)

        # User-requested metric style: n_A+B / (n_A+B + n_A) * 100
        pct_shared_given_a = 100.0 * n_ab / (n_ab + n_a) if (n_ab + n_a) else 0.0
        pct_shared_given_b = 100.0 * n_ab / (n_ab + n_b) if (n_ab + n_b) else 0.0

        pairwise[key_ab] = {
            "n_shared_clusters": n_ab,
            "pct_shared_given_" + lib_a: pct_shared_given_a,
            "pct_shared_given_" + lib_b: pct_shared_given_b,
        }

    return {
        "total_clusters": total_clusters,
        "flag_counts": flag_counts,
        "per_library": per_lib,
        "pairwise": pairwise,
    }


def compute_molecule_level_metrics(
    lib_totals: dict[str, int],
    exclusive_totals: dict[str, int],
    shared_totals: dict[str, int],
    cluster_presence_metrics: dict,
):
    """Compute molecule-level metrics for diversity/shared-space quantification."""
    out = {}
    for lib, n_total in lib_totals.items():
        n_excl = exclusive_totals.get(lib, 0)
        n_shared = shared_totals.get(lib, 0)
        n_clusters_with_lib = cluster_presence_metrics["per_library"][lib]["n_clusters_with_lib"]

        out[lib] = {
            "n_molecules_total": n_total,
            "n_molecules_exclusive": n_excl,
            "n_molecules_shared": n_shared,
            "pct_molecules_shared": 100.0 * n_shared / n_total if n_total else 0.0,
            "pct_molecules_exclusive": 100.0 * n_excl / n_total if n_total else 0.0,
            # Requested style metric: n clusters with lib / total molecules in lib
            "clusters_with_lib_over_total_molecules": n_clusters_with_lib / n_total if n_total else 0.0,
        }
    return out


def plot_overlap_sets(
    flag_counts: dict | Counter,
    lib_names: list[str],
    save_path: str | None = None,
    upset: bool = False,
):
    """Plot overlap as Venn (2-3 libs) or UpSet (recommended for many libs)."""
    venn_lib_comp(
        counts=dict(flag_counts),
        lib_names=lib_names,
        save_path=save_path,
        upset=upset,
    )


def plot_two_library_space(
    exclusive_totals: dict[str, int] | str | None = None,
    shared_totals: dict[str, int] | str | None = None,
    lib_a: str | None = None,
    lib_b: str | None = None,
    cluster_counts: list[Counter] | None = None,
    include_multi_combinations: bool = True,
    save_path: str | None = None,
):
    """Plot a pair-specific 2-library exclusive/shared molecule composition.

    Preferred mode:
        Pass ``cluster_counts`` (from ``count_and_map_clusters``) so the plot is
        specific to ``lib_a`` and ``lib_b`` even when 3+ libraries are present.

        - Exclusive ``lib_a``: molecules from clusters containing ``lib_a`` and
          not ``lib_b``.
        - Exclusive ``lib_b``: molecules from clusters containing ``lib_b`` and
          not ``lib_a``.
        - Shared space: clusters containing both ``lib_a`` and ``lib_b``.
          If ``include_multi_combinations=True``, higher-order intersections
          (e.g., ``A+B+C`` when plotting ``A`` vs ``B``) are included.

    Backward-compatible mode:
        If ``cluster_counts`` is not provided, the function falls back to
        ``exclusive_totals`` and ``shared_totals``.
    """
    # Allow new positional style: plot_two_library_space("A", "B", cluster_counts=...)
    if isinstance(exclusive_totals, str) and isinstance(shared_totals, str) and lib_a is None and lib_b is None:
        lib_a = exclusive_totals
        lib_b = shared_totals
        exclusive_totals = None
        shared_totals = None

    if lib_a is None or lib_b is None:
        raise ValueError("Both lib_a and lib_b must be provided")

    if lib_a == lib_b:
        raise ValueError("lib_a and lib_b must be different labels")

    if cluster_counts is not None:
        exclusive_a = 0
        exclusive_b = 0
        shared_a = 0
        shared_b = 0

        for cluster in cluster_counts:
            has_a = lib_a in cluster
            has_b = lib_b in cluster

            if has_a and has_b:
                if include_multi_combinations or len(cluster) == 2:
                    shared_a += cluster.get(lib_a, 0)
                    shared_b += cluster.get(lib_b, 0)
            elif has_a:
                exclusive_a += cluster.get(lib_a, 0)
            elif has_b:
                exclusive_b += cluster.get(lib_b, 0)
    else:
        if not isinstance(exclusive_totals, dict) or not isinstance(shared_totals, dict):
            raise ValueError(
                "Provide cluster_counts for pair-specific mode, or provide both "
                "exclusive_totals and shared_totals for backward-compatible mode."
            )
        exclusive_a = exclusive_totals.get(lib_a, 0)
        exclusive_b = exclusive_totals.get(lib_b, 0)
        shared_a = shared_totals.get(lib_a, 0)
        shared_b = shared_totals.get(lib_b, 0)

    x = [0, 1, 2]
    labels = [f"Exclusive {lib_a}", "Shared space", f"Exclusive {lib_b}"]

    plt.figure(figsize=(8, 5))
    plt.bar(x[0], exclusive_a, color="#4C78A8", label=f"From library {lib_a}")
    plt.bar(x[1], shared_a, color="#4C78A8")
    plt.bar(x[1], shared_b, bottom=shared_a, color="#F58518", label=f"From library {lib_b}")
    plt.bar(x[2], exclusive_b, color="#F58518")

    totals_for_labels = [exclusive_a, shared_a + shared_b, exclusive_b]
    offset = max(totals_for_labels) * 0.01 if totals_for_labels else 0.0

    plt.text(x[0], exclusive_a + offset, f"{exclusive_a:,}", ha="center", va="bottom")
    plt.text(x[2], exclusive_b + offset, f"{exclusive_b:,}", ha="center", va="bottom")

    # Keep A label outside lower shared segment for readability when it is very small.
    plt.text(
        x[1],
        shared_a + offset * 0.5,
        f"{lib_a}: {shared_a:,}",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )
    plt.text(
        x[1],
        shared_a + shared_b / 2,
        f"{lib_b}: {shared_b:,}",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )
    plt.text(x[1], shared_a + shared_b + offset, f"{shared_a + shared_b:,}", ha="center", va="bottom")

    plt.xticks(x, labels)
    plt.ylabel("Number of molecules")
    plt.title(f"Exclusive and Shared Chemical Space: {lib_a} vs {lib_b}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()
    plt.close("all")


def plot_multi_library_combination_space(
    cluster_counts: list[Counter],
    lib_order: list[str] | None = None,
    normalize: bool = False,
    save_path: str | None = None,
):
    """Plot stacked bars for all exact library-membership combinations.

    Each bar represents one exact combination label (e.g., ``A``, ``A+B``,
    ``A+B+C``). Stacks within a bar show molecule contributions by each library
    for that combination.

    Args:
        cluster_counts: Output from ``count_and_map_clusters``.
        lib_order: Optional explicit plotting order for library labels.
        normalize: If True, each bar is shown as percentages summing to 100.
        save_path: Optional path to save image. If omitted, plot is shown.

    Returns:
        dict with keys ``combo_labels`` and ``combo_totals``.
    """
    if not cluster_counts:
        raise ValueError("cluster_counts is empty")

    if lib_order is None:
        lib_order = sorted({lib for cluster in cluster_counts for lib in cluster.keys()})

    # Aggregate molecule counts by exact combination membership.
    combo_to_lib_counts: dict[str, Counter] = {}
    for cluster in cluster_counts:
        members = [lib for lib in sorted(cluster.keys()) if cluster.get(lib, 0) > 0]
        if not members:
            continue

        combo_label = "+".join(members)
        combo_to_lib_counts.setdefault(combo_label, Counter())
        for lib in members:
            combo_to_lib_counts[combo_label][lib] += cluster.get(lib, 0)

    if not combo_to_lib_counts:
        raise ValueError("No non-empty library combinations found in cluster_counts")

    combo_labels = sorted(combo_to_lib_counts.keys(), key=lambda x: (len(x.split("+")), x))
    combo_totals = [sum(combo_to_lib_counts[label].values()) for label in combo_labels]

    x = np.arange(len(combo_labels))
    width = 0.8

    plt.figure(figsize=(max(10, 0.8 * len(combo_labels)), 6))
    bottoms = np.zeros(len(combo_labels), dtype=float)

    cmap = plt.get_cmap("tab20")
    for i, lib in enumerate(lib_order):
        values = np.array([combo_to_lib_counts[label].get(lib, 0) for label in combo_labels], dtype=float)
        if normalize:
            totals = np.array(combo_totals, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(totals > 0, 100.0 * values / totals, 0.0)

        if np.any(values > 0):
            plt.bar(
                x,
                values,
                width,
                bottom=bottoms,
                label=lib,
                color=cmap(i % 20),
            )
            bottoms += values

    for i, total in enumerate(combo_totals):
        if normalize:
            label_y = 101.0
            text = f"n={int(total):,}"
        else:
            label_y = total + (0.01 * max(combo_totals) if combo_totals else 0.0)
            text = f"{int(total):,}"
        plt.text(i, label_y, text, ha="center", va="bottom", fontsize=8)

    plt.xticks(x, combo_labels, rotation=45, ha="right")
    plt.xlabel("Exact membership combination")
    plt.ylabel("Molecule percentage" if normalize else "Number of molecules")
    plt.title("Combination Membership Across Libraries")
    plt.legend(title="Library", ncol=min(4, max(1, len(lib_order))))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()
    plt.close("all")

    return {
        "combo_labels": combo_labels,
        "combo_totals": combo_totals,
    }


def load_smiles_by_library(smiles_dirs: dict[str, str | Path]) -> dict[str, list[str]]:
    """Load SMILES per library from directories of chunked .smi files."""
    return {lib: load_multiple_smiles(str(path)) for lib, path in smiles_dirs.items()}


def get_smiles_from_cluster_ids(
    mol_ids_cluster,
    smiles_by_library: dict[str, list[str]],
):
    """Extract SMILES and source labels from a cluster's molecule ids.

    Works with string ids and byte-buffer ids.
    """
    smiles_to_display = []
    labels = []

    for mol_id in mol_ids_cluster:
        normalized_id = _normalize_mol_id(mol_id)
        lib_label = _extract_lib_label(normalized_id)
        local_index = _extract_local_index(normalized_id)

        if lib_label not in smiles_by_library:
            raise KeyError(
                f"Library label '{lib_label}' from id '{normalized_id}' not found in smiles_by_library keys: {list(smiles_by_library.keys())}"
            )

        smiles_lib = smiles_by_library[lib_label]
        if local_index >= len(smiles_lib):
            raise IndexError(
                f"Index {local_index} out of range for library '{lib_label}' with {len(smiles_lib)} molecules"
            )

        smiles_to_display.append(smiles_lib[local_index])
        labels.append(lib_label)

    return smiles_to_display, labels


def sample_shared_cluster_indices(cluster_counts: list[Counter], n_samples: int = 5, random_state: int = 42):
    """Sample indices of clusters containing molecules from 2+ libraries."""
    shared_indices = [i for i, count in enumerate(cluster_counts) if len(count) > 1]
    if not shared_indices:
        return []

    rng = np.random.default_rng(random_state)
    n_pick = min(n_samples, len(shared_indices))
    selected = rng.choice(shared_indices, size=n_pick, replace=False)
    return selected.tolist()


def sample_cluster_indices_for_label(
    label: str,
    final_mol_ids=None,
    cluster_flags: list[str] | None = None,
    n_samples: int = 5,
    random_state: int = 42,
) -> list[int]:
    """Sample cluster indices for one exact label (e.g., 'A', 'A+B')."""
    indices = cluster_indices_for_label(label=label, final_mol_ids=final_mol_ids, cluster_flags=cluster_flags)
    if not indices:
        return []

    rng = np.random.default_rng(random_state)
    n_pick = min(n_samples, len(indices))
    selected = rng.choice(indices, size=n_pick, replace=False)
    return selected.tolist()


def _select_indices_with_required_labels(
    labels: list[str],
    max_mols: int,
    required_labels: list[str] | None,
    random_state: int = 42,
) -> list[int]:
    """Select molecule indices ensuring required labels are represented.

    If max_mols is smaller than number of required labels, priority is given in
    required_labels order.
    """
    n = len(labels)
    if n <= max_mols:
        return list(range(n))

    rng = np.random.default_rng(random_state)

    if not required_labels:
        return rng.choice(np.arange(n), size=max_mols, replace=False).tolist()

    selected: list[int] = []
    for req in required_labels:
        req_idx = [i for i, lb in enumerate(labels) if lb == req]
        if req_idx:
            selected.append(int(rng.choice(req_idx, size=1, replace=False)[0]))

    # Deduplicate while preserving order
    selected = list(dict.fromkeys(selected))

    if len(selected) >= max_mols:
        return selected[:max_mols]

    remaining = [i for i in range(n) if i not in selected]
    n_extra = max_mols - len(selected)
    extra = rng.choice(np.array(remaining), size=n_extra, replace=False).tolist()
    return selected + extra


def visualize_cluster_molecules(
    mol_ids_cluster,
    smiles_by_library: dict[str, list[str]],
    max_mols: int = 25,
    random_state: int = 42,
    mols_per_row: int = 5,
    required_labels: list[str] | None = None,
):
    """Render a molecule grid for one cluster with legends showing source library."""
    smiles_to_display, labels = get_smiles_from_cluster_ids(mol_ids_cluster, smiles_by_library)

    if len(smiles_to_display) > max_mols:
        idx = _select_indices_with_required_labels(
            labels=labels,
            max_mols=max_mols,
            required_labels=required_labels,
            random_state=random_state,
        )
        smiles_to_display = [smiles_to_display[i] for i in idx]
        labels = [labels[i] for i in idx]

    return smiles_to_grid_image(
        smiles_to_display,
        mols_per_row=mols_per_row,
        legends=labels,
    )


def visualize_shared_clusters(
    final_mol_ids,
    smiles_dirs: dict[str, str | Path],
    n_clusters: int = 3,
    max_mols_per_cluster: int = 25,
    random_state: int = 42,
    required_labels: list[str] | None = None,
    cluster_indices: list[int] | None = None,
):
    """Generate molecule-grid images for sampled shared clusters.

    Returns:
        list of dicts with keys: cluster_index, counts, image
    """
    if isinstance(final_mol_ids, dict):
        clusters = extract_mols_from_buffers(final_mol_ids)
    else:
        clusters = [[_normalize_mol_id(mol_id) for mol_id in cluster] for cluster in final_mol_ids]

    cluster_counts, cluster_flags = count_and_map_clusters(clusters)

    if cluster_indices is None:
        picked = sample_shared_cluster_indices(cluster_counts, n_samples=n_clusters, random_state=random_state)
    else:
        picked = [idx for idx in cluster_indices if 0 <= idx < len(clusters) and len(cluster_counts[idx]) > 1]

    smiles_by_library = load_smiles_by_library(smiles_dirs)
    if required_labels is None:
        required_labels = sorted(smiles_by_library.keys())

    out = []
    for idx in picked:
        img = visualize_cluster_molecules(
            mol_ids_cluster=clusters[idx],
            smiles_by_library=smiles_by_library,
            max_mols=max_mols_per_cluster,
            random_state=random_state,
            required_labels=required_labels,
        )
        out.append(
            {
                "cluster_index": idx,
                "flag": cluster_flags[idx],
                "counts": cluster_counts[idx],
                "image": img,
            }
        )

    return out


def visualize_exclusive_clusters(
    final_mol_ids,
    smiles_dirs: dict[str, str | Path],
    label: str,
    n_clusters: int = 3,
    max_mols_per_cluster: int = 25,
    random_state: int = 42,
    cluster_indices: list[int] | None = None,
):
    """Generate molecule-grid images for exclusive clusters of one label.

    Args:
        label: Exclusive label to visualize (e.g., 'A' or 'B').
    """
    if isinstance(final_mol_ids, dict):
        clusters = extract_mols_from_buffers(final_mol_ids)
    else:
        clusters = [[_normalize_mol_id(mol_id) for mol_id in cluster] for cluster in final_mol_ids]

    cluster_counts, cluster_flags = count_and_map_clusters(clusters)
    smiles_by_library = load_smiles_by_library(smiles_dirs)

    if cluster_indices is None:
        picked = sample_cluster_indices_for_label(
            label=label,
            cluster_flags=cluster_flags,
            n_samples=n_clusters,
            random_state=random_state,
        )
    else:
        valid = set(cluster_indices_for_label(label=label, cluster_flags=cluster_flags))
        picked = [idx for idx in cluster_indices if idx in valid]

    out = []
    for idx in picked:
        img = visualize_cluster_molecules(
            mol_ids_cluster=clusters[idx],
            smiles_by_library=smiles_by_library,
            max_mols=max_mols_per_cluster,
            random_state=random_state,
            required_labels=[label],
        )
        out.append(
            {
                "cluster_index": idx,
                "flag": cluster_flags[idx],
                "counts": cluster_counts[idx],
                "image": img,
            }
        )

    return out




