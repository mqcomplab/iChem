from .bitbirch import BitBirch
from .similarity import optimal_threshold
import numpy as np # type: ignore

def hierarchical_bitbirch(fingerprints,
                          threshold=None,
                          steps=5,
                          branching_factor=1024,
                          **bitbirch_kwargs):
    """Hierarchical clustering using BitBirch.
    It initially clusters the data and the reclusters the BF nodes hierarchically by decreasing the threshold.

    Parameters
    ----------
    fingerprints : np.ndarray
        Array of binary fingerprints.
    threshold : float, optional
        Similarity threshold for clustering. If None, it will be determined automatically.
    steps : int, optional
        Number of hierarchical steps to perform.

    Returns
    -------
    cluster_ids : dict
        A dictionary where keys are step indices and values are lists of clusters (each cluster is a list of molecule IDs).
    """

    if threshold is None:
        initial_threshold = optimal_threshold(fingerprints)
    else:
        initial_threshold = threshold

    cluster_ids = {}

    for k, threshold in enumerate(np.linspace(initial_threshold, 0, num=steps)):
        if threshold == initial_threshold:
            bb_object = BitBirch(threshold=threshold,
                                 merge_criterion='diameter',
                                 branching_factor=branching_factor)
            bb_object.fit(fingerprints)

            cluster_ids[k] = bb_object.get_cluster_mol_ids()
            print(f"Step {k}: Threshold = {bb_object.threshold:.3f}, Number of clusters = {len(cluster_ids[k])}")
        else:
            bb_object.recluster_inplace(iterations=1,
                                        extra_threshold=threshold - bb_object.threshold,
                                        shuffle=True)
            
            cluster_ids[k] = bb_object.get_cluster_mol_ids()
            # Print number of clusters at this step
            print(f"Step {k}: Threshold = {bb_object.threshold:.3f}, Number of clusters = {len(cluster_ids[k])}")

    return cluster_ids



