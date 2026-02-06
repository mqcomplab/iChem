import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import heatmap
#import plotly.graph_objects as go
from collections import Counter, defaultdict
from rdkit.Chem import rdFMCS, SaltRemover
from rdkit import Chem
from rdkit.Chem import Draw
import iChem.bblean.similarity as iSIM

def clusters_pop_plot(bitbirch_obj,
                      save_path: str = None,
                      ):

    """Plot the population distribution of clusters as a stacked bar chart.

    Args:
        bitbirch_obj: BitBirch clustering object with fitted clusters.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """

    # Calculate the counts of the populations
    populations = bitbirch_obj.get_cluster_populations(sort=True)
    n_1000 = sum(1 for pop in populations if pop > 1000)
    n_100 = sum(1 for pop in populations if pop > 100)
    n_10 = sum(1 for pop in populations if pop > 10)
    n_1 = sum(1 for pop in populations if pop > 1)
    n_0 = sum(1 for pop in populations if pop > 0)

    plt.figure(figsize=(3, 4))
    plt.bar('Num_cluster', n_0, label='>0', color='blue')
    plt.bar('Num_cluster', n_1, label='>1', color='orange')
    plt.bar('Num_cluster', n_10, label='>10', color='gray')
    plt.bar('Num_cluster', n_100, label='>100', color='green')
    plt.bar('Num_cluster', n_1000, label='>1000', color='red')
    plt.legend()
    plt.ylabel('Number of Clusters')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()


def clusters_pop_isim_plot(bitbirch_obj,
                           save_path: str = None,
                           figsize: tuple = (12, 6),
                           top=20,
                           initial=0):
    """Plot cluster population as bars with iSIM values on secondary axis.

    Args:
        bitbirch_obj: BitBirch clustering object with fitted clusters.
        save_path (str, optional): Path to save the plot. Defaults to None.
        figsize (tuple, optional): Figure size (width, height). Defaults to (12, 6).
        top (int, optional): Number of top clusters to display. Defaults to 20.
        initial (int, optional): Starting index for clusters to display. Defaults to 0."""

    # Get cluster populations and iSIM values
    all_populations = bitbirch_obj.get_cluster_populations(sort=True)
    isim_values = bitbirch_obj.get_iSIM_clusters(sort=True)
    
    # Calculate statistics before limiting
    total_clusters = len(all_populations)
    n_singletons = sum(1 for pop in all_populations if pop == 1)

    # Limit to top clusters for display
    populations = all_populations[initial:top]
    isim_values = isim_values[initial:top]

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot cluster populations as bars
    x = np.arange(len(populations))
    bars = ax1.bar(x, populations, alpha=0.7, color='blue', label='Population')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Population', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in x], rotation=45, ha='right')

    # Create secondary axis for iSIM values
    ax2 = ax1.twinx()
    line = ax2.plot(x, isim_values, color='darkorange', marker='o', 
                    linewidth=2, markersize=6, label='iSIM')
    ax2.set_ylabel('iSIM', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Add annotation with cluster statistics
    annotation_text = f'Total Clusters: {total_clusters}\nSingletons: {n_singletons}'
    ax1.text(0.98, 0.98, annotation_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.title('Cluster Population and iSIM')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()


def sampled_MSC(cluster,
                smiles,
                n_samples=100,
                MCS_threshold=0.75,
                return_samples=False):
    """Calculate the MSC for a sampled subset of molecules in a cluster."""
    n_samples = min(n_samples, len(cluster))

    # Get the SMILES for the sampled molecules
    remover = SaltRemover.SaltRemover()
    np.random.shuffle(cluster)
    sampled_smiles = [smiles[i] for i in cluster[:n_samples]]
    mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
    mols = [remover.StripMol(mol) for mol in mols]

    # Find the MSC of those molecules
    MCS = rdFMCS.FindMCS(mols, threshold=MCS_threshold)
    MCS_mol = Chem.MolFromSmarts(MCS.smartsString)

    if return_samples:
        return mols, MCS_mol
    
def sampled_MSC_image(cluster,
                        smiles,
                        n_samples=100,
                        MCS_threshold=0.75,
                        return_samples=False):
    
    if len(cluster) == 1:
        mol = Chem.MolFromSmiles(smiles[cluster[0]])
        return Draw.MolToImage(mol)

    if not return_samples:
        MCS_mol = sampled_MSC(cluster, smiles,
                            n_samples=n_samples,
                            MCS_threshold=MCS_threshold,
                            return_samples=False)
        return Draw.MolToImage(MCS_mol)
    
    mols, MCS_mol = sampled_MSC(cluster, smiles,
                                n_samples=n_samples,
                                MCS_threshold=MCS_threshold,
                                return_samples=True)

    # Generate the image of the MCS
    img1 = Draw.MolToImage(MCS_mol)

    # Find substructure matches in the sampled molecules with the MCS
    for mol in mols:
        if mol.HasSubstructMatch(MCS_mol):
            match = mol.GetSubstructMatch(MCS_mol)
            atom_indices = list(match)
            highlight_atoms = atom_indices
            # Highlight the matching substructure
            mol.SetProp('_highlightAtoms', ','.join(map(str, highlight_atoms)))

    highlight_lists = []
    for mol in mols:
        if mol.HasProp('_highlightAtoms'):
            vals = mol.GetProp('_highlightAtoms').split(',')
            highlight_lists.append(list(map(int, vals)))
        else:
            highlight_lists.append([])

    img2 = Draw.MolsToGridImage(mols,
                                highlightAtomLists=highlight_lists,
                                molsPerRow=5,
                                subImgSize=(250, 250))
    
    return img1, img2

def clusters_MSC(clusters: list,
                 smiles: list,
                 top_clusters: int = 10,
                 n_samples: int = 100,
                 MCS_threshold: float = 0.75,
                 save_path: str = None):
    """Generate and save/display MSC images for all clusters.

    Args:
        clusters (list): List of clusters (each cluster is a list of molecule indices).
        smiles (list): List of SMILES strings corresponding to the molecules.
        n_samples (int, optional): Number of molecules to sample from each cluster. Defaults to 100.
        MCS_threshold (float, optional): Threshold for MCS calculation. Defaults to 0.75.
        save_path (str, optional): Path to save the combined image. Defaults to None.
    """

    MCS_mols = []
    for cluster in clusters[:top_clusters]:
        MCS = sampled_MSC(cluster,
                           smiles,
                           n_samples=n_samples,
                           MCS_threshold=MCS_threshold,
                           return_samples=False)
        MCS_mols.append(MCS)

    img = Draw.MolsToGridImage(MCS_mols,
                                molsPerRow=5,
                                subImgSize=(250, 250),
                                legends=[f'Cluster {i+1}' for i in range(len(MCS_mols))])
    if save_path:
        img.save(save_path)
    else:
        return img
    
def cluster_mix_MCS_image(cluster,
                              smiles,
                              flags,
                              n_samples=50,
                              MCS_threshold=0.75,
                              save_path: str = None):
    """Generate MCS images for mixed clusters.

    Args:
        cluster (list): List of molecule indices in the cluster.
        smiles (list): List of SMILES strings corresponding to the molecules.
        n_samples (int, optional): Number of molecules to sample from the cluster. Defaults to 100."""
    
    n_samples = min(n_samples, len(cluster))
    if n_samples == 1:
        mol = Chem.MolFromSmiles(smiles[cluster[0]])
        return Draw.MolToImage(mol)

    # Get the SMILES for the sampled molecules
    remover = SaltRemover.SaltRemover()
    np.random.shuffle(cluster)
    sampled_smiles = [smiles[i] for i in cluster[:n_samples]]
    flags_sampled = [flags[i] for i in cluster[:n_samples]]
    mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
    mols = [remover.StripMol(mol) for mol in mols]

    # Find the MSC of those molecules
    MCS = rdFMCS.FindMCS(mols, threshold=MCS_threshold)
    MCS_mol = Chem.MolFromSmarts(MCS.smartsString)

    # Find substructure matches in the sampled molecules with the MCS
    for mol in mols:
        if mol.HasSubstructMatch(MCS_mol):
            match = mol.GetSubstructMatch(MCS_mol)
            atom_indices = list(match)
            highlight_atoms = atom_indices
            # Highlight the matching substructure
            mol.SetProp('_highlightAtoms', ','.join(map(str, highlight_atoms)))

    highlight_lists = []
    for mol in mols:
        if mol.HasProp('_highlightAtoms'):
            vals = mol.GetProp('_highlightAtoms').split(',')
            highlight_lists.append(list(map(int, vals)))
        else:
            highlight_lists.append([])

    img = Draw.MolsToGridImage(mols,
                                highlightAtomLists=highlight_lists,
                                molsPerRow=5,
                                subImgSize=(250, 250),
                                legends=flags_sampled)
    if save_path:
        img.save(save_path)
    else:
        return img

def pie_chart_mixed_clusters(counts: dict,
                             save_path: str = None):
    """Generate a pie chart of mixed cluster compositions.

    Args:
        counts (dict): Dictionary with cluster composition counts.
        save_path (str, optional): Path to save the pie chart. Defaults to None.
    """
    plt.figure(figsize=(10, 5))
    counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=False))
    plt.pie(counts.values(), labels=[None]*len(counts), autopct='%1.1f%%')
    plt.legend(labels=counts.keys(), loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

def symmetric_heatmap(results: list,
                      labels: list,
                      save_path: str = None,
                      only_upper: bool = True):
    """Generate a symmetric heatmap from a square results matrix.

    Args:
        results (list): 2D list or array of results.
    """
    # Do a heatmap wit only the upper triangle and the diagonal filled
    if only_upper:
        heatmap(np.array(results),
                xticklabels=labels, yticklabels=labels,
                annot=True, fmt='.2f', cmap='viridis', mask=np.tril(np.ones_like(results, dtype=bool), k=-1))
    else:
        heatmap(np.array(results),
                xticklabels=labels, yticklabels=labels,
                annot=True, fmt='.2f', cmap='viridis')
        
    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

def bar_chart_library_comparison(values: list[Counter],
                                 lib_names: list,
                                 save_path: str = None):
    """Generate a stacked bar chart showing population composition per cluster.

    Each element in `values` is a Counter mapping library label -> count.
    Bars are stacked so each library occupies a segment; x labels are 0..N-1.
    """
    if not values:
        return

    n_clusters = len(values)
    x = np.arange(n_clusters)

    # Choose a color palette with enough distinct colors
    colors = sns.color_palette("tab10", n_colors=max(10, len(lib_names)))
    colors = colors[:len(lib_names)]

    bottoms = np.zeros(n_clusters, dtype=float)
    plt.figure(figsize=(10, 5))

    # For each library, plot its segment on each cluster bar
    for lib, col in zip(lib_names, colors):
        heights = np.array([ctr.get(lib, 0) for ctr in values], dtype=float)
        plt.bar(x, heights, bottom=bottoms, color=col, label=lib)
        bottoms += heights

    plt.xticks(x, [str(i) for i in x], fontsize=8, rotation=45)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.yticks(np.arange(0, max(bottoms) + 1, step=max(1, max(bottoms) // 5)))
    plt.legend(title="Library", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

def venn_lib_comp(counts: dict,
                  lib_names: list = None,
                save_path: str = None):
    """Generate a Venn diagram showing library overlaps. Supports up to 3 libraries.

    Args:
        counts (dict): Dictionary with library overlap counts.
        save_path (str, optional): Path to save the Venn diagram. Defaults to None.

    Returns:
        fig: Matplotlib figure object.
    """

    from matplotlib_venn import venn2, venn3

    # Get total number of clusters for calculation of percentages
    total_clusters = sum(counts.values())

    # Sort the counts dictionary labels
    counts = dict(sorted(counts.items()))

    # Pass the counts to percentage with only one decimal place
    counts = {key: round((value / total_clusters) * 100, 1) for key, value in counts.items()}

    # Change the count labels to be used in venn diagrams
    new_counts = {}
    for key in counts.keys():
        if len(key.split('+')) == 3:
            new_counts["111"] = counts[key]
        elif len(key.split('+')) == 1:
            if key == lib_names[0]:
                new_counts["100"] = counts[key]
            elif key == lib_names[1]:
                new_counts["010"] = counts[key]
            elif key == lib_names[2]:
                new_counts["001"] = counts[key]
        elif len(key.split('+')) == 2:
            libs = key.split('+')
            if lib_names[0] in libs and lib_names[1] in libs:
                new_counts["110"] = counts[key]
            elif lib_names[0] in libs and lib_names[2] in libs:
                new_counts["101"] = counts[key]
            elif lib_names[1] in libs and lib_names[2] in libs:
                new_counts["011"] = counts[key]

    # Plot the venn diagram
    plt.figure(figsize=(5, 5))
    if len(lib_names) == 2:
        venn2(subsets=(new_counts.get("100", 0),
                       new_counts.get("010", 0),
                       new_counts.get("110", 0)),
              set_labels=(f"{lib_names[0]}", f"{lib_names[1]}"),
              set_colors=('blue', 'orange', 'green'),
              alpha=0.75)
    elif len(lib_names) == 3:
        venn3(subsets=(new_counts.get("100", 0),
                       new_counts.get("010", 0),
                       new_counts.get("110", 0),
                       new_counts.get("001", 0),
                       new_counts.get("101", 0),
                       new_counts.get("011", 0),
                       new_counts.get("111", 0)),
              set_labels=(f"{lib_names[0]}", f"{lib_names[1]}", f"{lib_names[2]}"),
                set_colors=('blue', 'orange', 'green'),
                alpha=0.75)
    else:
        raise ValueError("Only 2 or 3 libraries are supported for Venn diagrams.")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()
