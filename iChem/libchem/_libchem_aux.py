import numpy as np # type: ignore
from ..bblean.similarity import jt_isim_packed
from collections import Counter

def interiSIM(fps1: np.ndarray,
              fps2: np.ndarray) -> float:
    """Calculate the inter-library iSIM between two sets of packed fingerprints.

    Args:
        fps1 (np.ndarray): Packed fingerprints of the first library.
        fps2 (np.ndarray): Packed fingerprints of the second library.

    Returns:
        float: The inter-library iSIM value.
    """
    n1 = len(fps1)
    n2 = len(fps2)
    n3 = n1 + n2

    iSIM_1 = jt_isim_packed(np.array(fps1))

    iSIM_2 = jt_isim_packed(np.array(fps2))

    iSIM_3 = jt_isim_packed(
        np.vstack((fps1, fps2)),
    )

    interiSIM = (iSIM_3 * n3 * (n3 - 1) - iSIM_1 * n1 * (n1 - 1) - iSIM_2 * n2 * (n2 - 1)) / (2 * n1 * n2)

    return interiSIM

def intraiSIM(fps1: np.ndarray,
              fps2: np.ndarray) -> float:
    """Calculate the intra-library iSIM for a set of packed fingerprints.

    Args:
        fps1 (np.ndarray): Packed fingerprints of the first library.
        fps2 (np.ndarray): Packed fingerprints of the second library.
    Returns:
        float: The intra-library iSIM value.
    """

    isim_value = jt_isim_packed(
                np.vstack((fps1, fps2))
            )
    
    return isim_value

def MaxSum(similarityMatrix: np.ndarray) -> float:
    """Determine the MaxSum cluster medoids from a similarity matrix.

    Args:
        SimilarityMatrix (np.ndarray): A square matrix of pairwise similarities.
    
    Returns:
        MaxSum float: The MaxSum value. Average max sim by column and row.
    """
    total_similarity = 0.0
    for i in range(similarityMatrix.shape[0]):
        max_sim = np.max(similarityMatrix[i, :])
        total_similarity += max_sim
    for k in range(similarityMatrix.shape[1]):
        max_sim = np.max(similarityMatrix[:, k])
        total_similarity += max_sim
    medoids_maxsum = total_similarity / (similarityMatrix.shape[0] + similarityMatrix.shape[1])

    return medoids_maxsum

def MinSum(similarityMatrix: np.ndarray) -> float:
    """Determine the MinSum cluster medoids from a similarity matrix.

    Args:
        SimilarityMatrix (np.ndarray): A square matrix of pairwise similarities.

    Returns:
        MinSum float: The MinSum value. Average min sim by column and row.
    """
    total_similarity = 0.0
    for i in range(similarityMatrix.shape[0]):
        min_sim = np.min(similarityMatrix[i, :])
        total_similarity += min_sim
    for k in range(similarityMatrix.shape[1]):
        min_sim = np.min(similarityMatrix[:, k])
        total_similarity += min_sim
    medoids_minsum = total_similarity / (similarityMatrix.shape[0] + similarityMatrix.shape[1])

    return medoids_minsum


def combo_counts(flags: list, library_names: list[str] | None = None) -> tuple[dict, dict]:
    """Return a mapping of library-combination -> list of cluster indices, and counts.

    Parameters
    ----------
    flags : list
        Iterable of per-cluster iterables/lists containing library-name flags
        for each member of the cluster (e.g., ['libA', 'libA', 'libB']).
    library_names : list[str] | None
        Optional list of known library names to ensure keys appear in the
        returned mapping even if empty. If None, the set of libraries
        found in `flags` is used.

    Returns
    -------
    mapping : dict
        Dictionary where keys are either a single-library name (for pure
        clusters) or a '+'-joined string of sorted library names for mixed
        combinations (e.g., 'libA+libB'), and values are lists of cluster
        indices (ints) that belong to that class.
    counts : dict
        Dictionary with the same keys as mapping where values are integer
        counts (lengths of the lists). Also includes a top-level 'mixed'
        key with the total number of mixed clusters.
    """
    mapping: dict = {}
    for idx, cluster_flags in enumerate(flags):
        # Normalize to a sorted tuple of unique library names
        unique_libs = tuple(sorted(set(cluster_flags)))
        if len(unique_libs) == 1:
            key = unique_libs[0]
        else:
            key = "+".join(unique_libs)
        mapping.setdefault(key, []).append(idx)

    # Ensure single-library keys are present if requested
    if library_names is not None:
        for lib in library_names:
            mapping.setdefault(lib, [])

    # Build counts dict from mapping
    counts = {key: len(idxs) for key, idxs in mapping.items()}
    # Ensure all requested single-library keys exist in counts
    if library_names is not None:
        for lib in library_names:
            counts.setdefault(lib, 0)

    return counts, mapping