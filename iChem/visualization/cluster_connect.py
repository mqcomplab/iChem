import numpy as np # type : ignore
from iChem.utils import binary_fps
from iChem.bblean.similarity import optimal_threshold
from iChem.bblean import BitBirch


def cluster_connect_vis(smiles: list[str],
                    fp_type: str = 'ECFP4',
                    n_bits: int = 2048,
                    fingerprints: np.ndarray = None,
                    initial_threshold: float = None,
                    steps: int = 5,
                    branching_factor: int = 1024,
                    min_size: int = 100,
                    ):
    """Cluster connection visualization to check how clusters are connected in chemical space
    at different thresholds.
    
    Parameters
    ----------
    smiles : list
        List of SMILES strings.
    fp_type : str, optional
        Type of fingerprint to use. Default is 'ECFP4'.
    fingerprints : np.ndarray, optional
        Precomputed fingerprints. If provided, `smiles` will be ignored.
    n_bits : int, optional
        Number of bits for the fingerprint. Default is 2048.
    initial_threshold : float, optional
        Initial threshold for clustering. If None, it will be estimated.
    steps : int, optional
        Number of threshold steps to visualize. Default is 5.
    branching_factor : int, optional
        Branching factor for the BitBirch clustering. Default is 1024.
    min_size : int, optional
        Minimum cluster size to consider. Default is 100.

    Returns
    -------
    Visualization of cluster connections at different thresholds.
    """
    
    if fingerprints is None:
        fingerprints, invalid_smiles = binary_fps(smiles=smiles,
                                  fp_type=fp_type,
                                  n_bits=n_bits,
                                  packed=True,
                                  return_invalid=True,
                                  )
        
        if invalid_smiles:
            print(f"Warning: The following SMILES strings are invalid and will be ignored: \n {np.array(smiles)[invalid_smiles]}")

            smiles = [s for i, s in enumerate(smiles) if i not in invalid_smiles]

    if len(fingerprints) != len(smiles):
        raise ValueError("Length of fingerprints does not match length of SMILES strings.")
    

    # Perform the initial clustering
    if initial_threshold is None:
        initial_threshold = optimal_threshold(fingerprints)


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

    # Visualization
    # Do a Sankey diagram to see how clusters are connected at different thresholds
    try:
        import plotly.graph_objects as go # type: ignore
    except ImportError:
        raise ImportError("plotly is required for visualization. Please install it via 'pip install plotly'.")
    
    # Build Sankey diagram with aggregated flows
    labels = []
    sources = []
    targets = []
    values = []
    node_customdata = []  # (cluster label, molecule count)
    
    # Create labels for each cluster at each step
    node_idx = 0
    step_cluster_to_node = {}  # Maps (step, cluster_idx) to node index
    
    for step in range(steps):
        clusters = cluster_ids[step]
        step_cluster_to_node[step] = {}
        small_cluster_count = 0
        
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) > min_size:
                labels.append(f"")
                node_customdata.append((f"{cluster_idx}", len(cluster)))
                step_cluster_to_node[step][cluster_idx] = node_idx
                node_idx += 1
            else:
                small_cluster_count += len(cluster)
        
        # Add a node for small clusters if any exist
        if small_cluster_count > 0:
            labels.append(f"Small Clusters")
            node_customdata.append(("Small", small_cluster_count))
            step_cluster_to_node[step]['small'] = node_idx
            node_idx += 1
    
    # Build aggregated connections between consecutive steps
    for step in range(steps - 1):
        current_clusters = cluster_ids[step]
        next_clusters = cluster_ids[step + 1]

        # Create a mapping of current cluster indices to next cluster indices
        flow_map = {}

        for curr_cluster_idx, curr_cluster in enumerate(current_clusters):
            source_label = 'small' if len(curr_cluster) <= min_size else curr_cluster_idx
            # Skip if the source node does not exist (e.g., no small clusters at this step)
            if source_label not in step_cluster_to_node[step]:
                continue

            curr_mol_set = set(curr_cluster)
            dominant_next = None
            max_overlap = 0

            for next_cluster_idx, next_cluster in enumerate(next_clusters):
                overlap = len(curr_mol_set.intersection(set(next_cluster)))
                if overlap > max_overlap:
                    max_overlap = overlap
                    target_label = next_cluster_idx if len(next_cluster) > min_size else 'small'
                    # Only keep target if it exists (handles cases with no small node at next step)
                    if target_label in step_cluster_to_node[step + 1]:
                        dominant_next = target_label

            if dominant_next is not None and max_overlap > 0:
                key = (
                    step_cluster_to_node[step][source_label],
                    step_cluster_to_node[step + 1][dominant_next],
                )
                flow_map[key] = flow_map.get(key, 0) + max_overlap

        # Add aggregated links
        for (source, target), count in flow_map.items():
            sources.append(source)
            targets.append(target)
            values.append(count)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            customdata=node_customdata,
            hovertemplate="Cluster %{customdata[0]}<br>%{customdata[1]} molecules<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            hovertemplate="%{value} molecules<extra></extra>",
        )
    )])
    
    fig.update_layout(title_text="Cluster Connection Visualization", font_size=10)
    fig.show()