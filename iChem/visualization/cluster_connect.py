import numpy as np # type: ignore
import base64
import json
import tempfile
import os
from iChem.utils import binary_fps, count_fps, real_fps, normalize_fps, minmax_norm
from iChem.bblean.similarity import optimal_threshold
from iChem.bblean.hierarchical import hierarchical_bitbirch
from iChem.bbreal.hierarchical import hierarchical_bbreal
from iChem.bbreal import optimal_threshold_real
from .mol_images import smiles_to_grid_image


def cluster_connections(smiles: list[str],
                    fp_type: str = 'ECFP4',
                    n_bits: int = 2048,
                    fingerprints: np.ndarray = None,
                    initial_threshold: float = None,
                    steps: int = 5,
                    branching_factor: int = 1024,
                    min_size: int = 100,
                    clustering_fp_type: str = "binary",
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
    clustering_fp_type : str, optional
        Type of fingerprint to use for clustering. Either "binary", "real" or "count". Default is "binary".

    Returns
    -------
    Visualization of cluster connections at different thresholds.
    """
    
    if fingerprints is None:
        if clustering_fp_type == "binary":
            fingerprints, invalid_smiles = binary_fps(smiles=smiles,
                                      fp_type=fp_type,
                                      n_bits=n_bits,
                                  packed=True,
                                  return_invalid=True,
                                  )
        elif clustering_fp_type == "real":
            fingerprints, invalid_smiles = real_fps(smiles=smiles,
                                  return_invalid=True,
                                  )
            
            # Normalize the fingerprints
            fingerprints = minmax_norm(fingerprints)
        elif clustering_fp_type == "count":
            fingerprints, invalid_smiles = count_fps(smiles=smiles,
                                      fp_type=fp_type,
                                      n_bits=n_bits,
                                  return_invalid=True,
                                  )
            
            # Normalize the fingerprints
            fingerprints = normalize_fps(fingerprints)
        else:
            raise ValueError(f"Invalid clustering_fp_type: {clustering_fp_type}. Must be 'binary', 'real' or 'count'.")

        if invalid_smiles:
            print(f"Warning: The following SMILES strings are invalid and will be ignored: \n {np.array(smiles)[invalid_smiles]}")

            smiles = [s for i, s in enumerate(smiles) if i not in invalid_smiles]

    if len(fingerprints) != len(smiles):
        raise ValueError("Length of fingerprints does not match length of SMILES strings.")
    

    # Perform the initial clustering
    if initial_threshold is None and clustering_fp_type == "binary":
        initial_threshold = optimal_threshold(fingerprints)
    if initial_threshold is None and (clustering_fp_type == "real" or clustering_fp_type == "count"):
        initial_threshold = optimal_threshold_real(fingerprints)


    if clustering_fp_type == "binary":
        cluster_ids = hierarchical_bitbirch(fingerprints,
                                        threshold=initial_threshold,
                                        steps=steps,
                                        branching_factor=branching_factor,
                                        )
    if clustering_fp_type == "real" or clustering_fp_type == "count":
        cluster_ids = hierarchical_bbreal(fingerprints,
                                        threshold=initial_threshold,
                                        steps=steps,
                                        branching_factor=branching_factor,
                                        )

    # Visualization
    # Do a Sankey diagram to see how clusters are connected at different thresholds
    try:
        import plotly.graph_objects as go # type: ignore
    except ImportError:
        raise ImportError("plotly is required for visualization. Please install it via 'pip install plotly'.")
    
    # Create a temporary directory for cluster images
    temp_dir = tempfile.mkdtemp(prefix="ichem_clusters_")
    print(f"Saving cluster images to: {temp_dir}")
    
    # Build Sankey diagram with aggregated flows
    labels = []
    sources = []
    targets = []
    values = []
    node_image_paths = {}  # Maps node index to image file path
    node_image_b64 = {}  # Maps node index to base64 data URL
    
    # Create labels for each cluster at each step
    node_idx = 0
    step_cluster_to_node = {}  # Maps (step, cluster_idx) to node index
    
    for step in range(steps):
        clusters = cluster_ids[step]
        step_cluster_to_node[step] = {}
        small_cluster_count = 0
        small_cluster_indices = []
        
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) > min_size:
                labels.append(f"")
                # Generate and save image for this cluster
                try:
                    # Get 5 random SMILES from the cluster
                    sampled_indices = np.random.choice(cluster, size=min(5, len(cluster)), replace=False)
                    sampled_smiles = [smiles[i] for i in sampled_indices]
                    img = smiles_to_grid_image(
                        smiles=sampled_smiles,
                        mols_per_row=5,
                        sub_img_size=(250, 250)
                    )
                    # Save image to temp directory
                    img_path = os.path.join(temp_dir, f"cluster_step{step}_c{cluster_idx}.png")
                    # Handle both PIL Image and byte data
                    if hasattr(img, 'save'):
                        img.save(img_path)
                    elif hasattr(img, 'data'):
                        # RDKit Image object with data attribute
                        with open(img_path, 'wb') as f:
                            f.write(img.data)
                    else:
                        raise ValueError(f"Unknown image type: {type(img)}")
                    
                    # Convert to base64 for display
                    with open(img_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                        node_image_b64[node_idx] = f"data:image/png;base64,{img_data}"
                    node_image_paths[node_idx] = img_path
                except Exception as e:
                    print(f"Warning: Could not generate image for cluster {cluster_idx} at step {step}: {e}")
                
                step_cluster_to_node[step][cluster_idx] = node_idx
                node_idx += 1
            else:
                small_cluster_count += len(cluster)
                small_cluster_indices.extend(cluster)
        
        # Add a node for small clusters if any exist
        if small_cluster_count > 0:
            labels.append(f"Small Clusters")
            step_cluster_to_node[step]['small'] = node_idx
            node_idx += 1
    
    # Build aggregated connections between consecutive steps
    link_image_b64 = {}  # Maps link index to base64 data URL
    link_idx = 0
    
    for step in range(steps - 1):
        current_clusters = cluster_ids[step]
        next_clusters = cluster_ids[step + 1]

        # Create a mapping of current cluster indices to next cluster indices
        flow_map = {}
        flow_molecules = {}  # Track molecules in each flow from small clusters

        for curr_cluster_idx, curr_cluster in enumerate(current_clusters):
            source_label = 'small' if len(curr_cluster) <= min_size else curr_cluster_idx
            # Skip if the source node does not exist (e.g., no small clusters at this step)
            if source_label not in step_cluster_to_node[step]:
                continue

            curr_mol_set = set(curr_cluster)
            dominant_next = None
            max_overlap = 0
            overlap_molecules = []

            for next_cluster_idx, next_cluster in enumerate(next_clusters):
                overlap_mols = curr_mol_set.intersection(set(next_cluster))
                overlap = len(overlap_mols)
                if overlap > max_overlap:
                    max_overlap = overlap
                    overlap_molecules = list(overlap_mols)
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
                
                # If this flow is from small clusters, store the molecules
                if source_label == 'small' and overlap_molecules:
                    if key not in flow_molecules:
                        flow_molecules[key] = []
                    flow_molecules[key].extend(overlap_molecules)

        # Add aggregated links and generate images for small cluster flows
        for (source, target), count in flow_map.items():
            sources.append(source)
            targets.append(target)
            values.append(count)
            
            # Generate image if this is a flow from small clusters
            if (source, target) in flow_molecules and flow_molecules[(source, target)]:
                try:
                    # Sample up to 5 molecules from the flow
                    flow_mols = flow_molecules[(source, target)]
                    sampled_indices = np.random.choice(flow_mols, size=min(5, len(flow_mols)), replace=False)
                    sampled_smiles = [smiles[i] for i in sampled_indices]
                    img = smiles_to_grid_image(
                        smiles=sampled_smiles,
                        mols_per_row=5,
                        sub_img_size=(250, 250)
                    )
                    # Save image to temp directory
                    img_path = os.path.join(temp_dir, f"flow_step{step}_s{source}_t{target}.png")
                    # Handle both PIL Image and byte data
                    if hasattr(img, 'save'):
                        img.save(img_path)
                    elif hasattr(img, 'data'):
                        with open(img_path, 'wb') as f:
                            f.write(img.data)
                    else:
                        raise ValueError(f"Unknown image type: {type(img)}")
                    
                    # Convert to base64 for display
                    with open(img_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                        link_image_b64[link_idx] = f"data:image/png;base64,{img_data}"
                except Exception as e:
                    print(f"Warning: Could not generate image for flow {source}->{target}: {e}")
            
            link_idx += 1
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            hovertemplate="%{value} molecules<extra></extra>",
        )
    )])
    
    fig.update_layout(
        title_text="Cluster Connection Visualization<br><sub>Hover over cluster nodes to view sample molecules</sub>", 
        font_size=10,
        height=700
    )
    
    # Display the figure with custom HTML for image display on hover
    try:
        from IPython.display import HTML, display
        
        fig_html = fig.to_html(include_plotlyjs='cdn', div_id='sankey-plot')
        
        # Create JavaScript to handle hover and display images as base64 data URLs
        script = f'''
        <script>
        var nodeImageB64 = {json.dumps(node_image_b64)};
        var linkImageB64 = {json.dumps(link_image_b64)};
        
        setTimeout(function() {{
            var plotDiv = document.getElementById('sankey-plot');
            if (plotDiv && plotDiv.data) {{
                plotDiv.on('plotly_hover', function(data) {{
                    var img = document.getElementById('cluster-img-display');
                    if (!img) return;
                    
                    if (data.points && data.points[0]) {{
                        var point = data.points[0];
                        
                        // Check if hovering over a node
                        if (point.pointNumber !== undefined && nodeImageB64[point.pointNumber]) {{
                            img.src = nodeImageB64[point.pointNumber];
                            img.style.display = 'block';
                        }}
                        // Check if hovering over a link
                        else if (point.index !== undefined && linkImageB64[point.index]) {{
                            img.src = linkImageB64[point.index];
                            img.style.display = 'block';
                        }}
                        else {{
                            img.style.display = 'none';
                        }}
                    }}
                }});
                
                plotDiv.on('plotly_unhover', function() {{
                    var img = document.getElementById('cluster-img-display');
                    if (img) {{
                        img.style.display = 'none';
                    }}
                }});
            }}
        }}, 100);
        </script>
        '''
        
        html = f'''
        <div>
            {fig_html}
            <div style="margin-top: 20px; text-align: center;">
                <div style="margin: 20px auto; max-width: 100%;">
                    <img id="cluster-img-display" style="max-width: 100%; max-height: 600px; display: none; border: 1px solid #ddd; padding: 10px;" />
                </div>
            </div>
        </div>
        {script}
        '''
        
        display(HTML(html))
    except ImportError:
        # Fallback for non-Jupyter environments
        print(f"Cluster images saved to: {temp_dir}")
        print("Note: Image display on hover works best in Jupyter notebooks.")
        fig.show()


def cluster_dendrogram(smiles: list[str],
                    fp_type: str = 'ECFP4',
                    n_bits: int = 2048,
                    fingerprints: np.ndarray = None,
                    initial_threshold: float = None,
                    steps: int = 5,
                    branching_factor: int = 1024,
                    min_size: int = 100,
                    ):
    """Cluster dendrogram visualization showing molecule indices on hover.
    
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
    Interactive dendrogram visualization with molecule indices on hover.
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


    cluster_ids = hierarchical_bitbirch(fingerprints,
                                        threshold=initial_threshold,
                                        steps=steps,
                                        branching_factor=branching_factor,
                                        )
    
    # Visualization
    # Create a Sankey diagram with molecule indices shown on hover
    try:
        import plotly.graph_objects as go # type: ignore
    except ImportError:
        raise ImportError("plotly is required for visualization. Please install it via 'pip install plotly'.")
    
    # Build Sankey diagram with aggregated flows and molecule indices
    labels = []
    sources = []
    targets = []
    values = []
    node_mol_indices = {}  # Maps node index to list of molecule indices
    link_mol_indices = {}  # Maps link index to list of molecule indices
    
    # Create labels for each cluster at each step
    node_idx = 0
    step_cluster_to_node = {}  # Maps (step, cluster_idx) to node index
    
    for step in range(steps):
        clusters = cluster_ids[step]
        step_cluster_to_node[step] = {}
        small_cluster_count = 0
        small_cluster_indices = []
        
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) > min_size:
                labels.append(f"Step {step} - C{cluster_idx}")
                node_mol_indices[node_idx] = cluster.tolist() if isinstance(cluster, np.ndarray) else list(cluster)
                step_cluster_to_node[step][cluster_idx] = node_idx
                node_idx += 1
            else:
                small_cluster_count += len(cluster)
                small_cluster_indices.extend(cluster)
        
        # Add a node for small clusters if any exist
        if small_cluster_count > 0:
            labels.append(f"Step {step} - Small ({small_cluster_count})")
            node_mol_indices[node_idx] = small_cluster_indices
            step_cluster_to_node[step]['small'] = node_idx
            node_idx += 1
    
    # Build aggregated connections between consecutive steps
    link_idx = 0
    
    for step in range(steps - 1):
        current_clusters = cluster_ids[step]
        next_clusters = cluster_ids[step + 1]

        # Create a mapping of current cluster indices to next cluster indices
        flow_map = {}
        flow_molecules = {}  # Track molecules in each flow

        for curr_cluster_idx, curr_cluster in enumerate(current_clusters):
            source_label = 'small' if len(curr_cluster) <= min_size else curr_cluster_idx
            # Skip if the source node does not exist
            if source_label not in step_cluster_to_node[step]:
                continue

            curr_mol_set = set(curr_cluster)
            dominant_next = None
            max_overlap = 0
            overlap_molecules = []

            for next_cluster_idx, next_cluster in enumerate(next_clusters):
                overlap_mols = curr_mol_set.intersection(set(next_cluster))
                overlap = len(overlap_mols)
                if overlap > max_overlap:
                    max_overlap = overlap
                    overlap_molecules = list(overlap_mols)
                    target_label = next_cluster_idx if len(next_cluster) > min_size else 'small'
                    # Only keep target if it exists
                    if target_label in step_cluster_to_node[step + 1]:
                        dominant_next = target_label

            if dominant_next is not None and max_overlap > 0:
                key = (
                    step_cluster_to_node[step][source_label],
                    step_cluster_to_node[step + 1][dominant_next],
                )
                flow_map[key] = flow_map.get(key, 0) + max_overlap
                
                # Store the molecules for this flow
                if key not in flow_molecules:
                    flow_molecules[key] = []
                flow_molecules[key].extend(overlap_molecules)

        # Add aggregated links
        for (source, target), count in flow_map.items():
            sources.append(source)
            targets.append(target)
            values.append(count)
            
            # Store molecule indices for this link
            if (source, target) in flow_molecules:
                link_mol_indices[link_idx] = flow_molecules[(source, target)]
            
            link_idx += 1
    
    # Create custom hover text with molecule indices
    node_hover_text = []
    for i, label in enumerate(labels):
        mol_indices = node_mol_indices.get(i, [])
        hover_text = f"{label}<br>Molecules: {mol_indices}"
        node_hover_text.append(hover_text)
    
    link_hover_text = []
    for i in range(len(sources)):
        mol_indices = link_mol_indices.get(i, [])
        hover_text = f"{len(mol_indices)} molecules<br>Indices: {mol_indices}"
        link_hover_text.append(hover_text)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            customdata=node_hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            customdata=link_hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )])
    
    fig.update_layout(
        title_text="Cluster Dendrogram<br><sub>Hover over nodes and links to view molecule indices</sub>", 
        font_size=10,
        height=700
    )
    
    # Display the figure
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        # Fallback for non-Jupyter environments
        fig.show()
    