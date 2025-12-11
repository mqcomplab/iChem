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

def clusters_pop_plot(clusters: list,
                      save_path: str = None,
                      ):

    """Plot the population of each cluster as a bar chart.

    Args:
        clusters (list): List of cluster sizes.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """

    # Calculate the counts of the populations
    lenghts = [len(c) for c in clusters]
    n_1000 = sum(1 for lenght in lenghts if lenght > 1000)
    n_100 = sum(1 for lenght in lenghts if lenght > 100)
    n_10 = sum(1 for lenght in lenghts if lenght > 10)
    n_1 = sum(1 for lenght in lenghts if lenght > 1)
    n_0 = sum(1 for lenght in lenghts if lenght > 0)

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
    