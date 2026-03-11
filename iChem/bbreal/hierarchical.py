from .bbreal import BBReal
from ._bbreal import optimal_threshold_real
import numpy as np  # type: ignore


def hierarchical_bbreal(X,
                        threshold=None,
                        steps=5,
                        branching_factor=1024,
                        merge_criterion='diameter',
                        return_threshold=False,
                        verbose=False,
                        **bbreal_kwargs):
    """Hierarchical clustering using BBReal.
    
    It initially clusters the data and then reclusters hierarchically by 
    decreasing the threshold, producing a hierarchical clustering structure.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data to cluster.
        
    threshold : float, optional
        Initial similarity threshold for clustering. If None, a default value
        of 0.5 will be used (BBReal default).
        
    steps : int, default=5
        Number of hierarchical clustering steps to perform. The threshold will
        be linearly decreased from the initial value to 0 over these steps.
        
    branching_factor : int, default=1024
        Maximum number of subclusters in each node of the BIRCH CF-tree.
        
    merge_criterion : str, default='diameter'
        Merge criterion to use. Currently only 'diameter' is supported.
        
    return_threshold : bool, default=False
        If True, also return the initial threshold used.
        
    verbose : bool, default=False
        If True, print progress information during clustering.
        
    **bbreal_kwargs
        Additional keyword arguments to pass to BBReal constructor.
    
    Returns
    -------
    cluster_ids : dict
        A dictionary where keys are step indices (0 to steps-1) and values
        are lists of molecule ID lists, one for each cluster at that step.
        Each entry represents the clustering at that hierarchy level.
        
    threshold : float (optional)
        The initial threshold used. Only returned if return_threshold=True.
    
    Notes
    -----
    - The first step (k=0) performs initial clustering at the highest threshold.
    - Subsequent steps decrease the threshold, allowing singleton clusters and
      tight clusters to potentially merge with others.
    - The final step uses threshold=0, which is the most permissive setting.
    - Shuffling is enabled during reclustering to reduce order-dependent artifacts.
    
    Examples
    --------
    >>> import numpy as np
    >>> from bbreal.hierarchical_bbreal import hierarchical_bbreal
    >>> X = np.random.rand(100, 10)
    >>> cluster_ids, init_threshold = hierarchical_bbreal(X, threshold=0.7, steps=5, return_threshold=True)
    >>> print(f"Step 0 has {len(cluster_ids[0])} clusters")
    >>> print(f"Step 4 has {len(cluster_ids[4])} clusters")
    """
    
    # Set initial threshold
    if threshold is None:
        initial_threshold = optimal_threshold_real(X)  # BBReal default
    else:
        initial_threshold = threshold
    
    cluster_ids = {}
    
    # Generate linearly spaced thresholds from initial to 0
    thresholds = np.linspace(initial_threshold, 0, num=steps)
    
    for k, current_threshold in enumerate(thresholds):
        if k == 0:
            # First step: initial clustering at highest threshold
            bb_object = BBReal(
                threshold=current_threshold,
                branching_factor=branching_factor,
                merge_criterion=merge_criterion
            )
            bb_object.fit(X)
            
            if verbose:
                print(f"Step {k}: threshold={current_threshold:.4f}, "
                      f"n_clusters={len(bb_object.get_BFs())}")
            
            cluster_ids[k] = bb_object.get_cluster_mol_ids()
        else:
            # Subsequent steps: recluster with decreased threshold
            threshold_change = current_threshold - bb_object.threshold
            
            bb_object.recluster_inplace(
                iterations=1,
                extra_threshold=threshold_change,
                shuffle=True,
                seed=42,
                verbose=False
            )
            
            if verbose:
                print(f"Step {k}: threshold={current_threshold:.4f}, "
                      f"n_clusters={len(bb_object.get_BFs())}")
            
            cluster_ids[k] = bb_object.get_cluster_mol_ids()
    
    if return_threshold:
        return cluster_ids, initial_threshold
    
    return cluster_ids
