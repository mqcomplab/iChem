"""Main ClusterAnalyzer class - Phase 1 Core Infrastructure."""

import numpy as np
from typing import Optional, List, Dict, Any
import pandas as pd #type : ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore

from bblean.similarity import jt_isim, jt_sim_packed #type: ignore

from ..utils import binary_fps
from ..visualization.mol_images import smiles_to_grid_image, MCS_image #type: ignore
from rdkit import Chem #type: ignore


class ClusterAnalyzer:
    """Comprehensive cluster analysis tool for clustering evaluation and metrics.
    
    Provides population statistics, diversity metrics, similarity analysis,
    and quality assessment for molecule clustering results.
    
    Attributes:
        clusters: List of lists with molecule indices (required)
        fps: Optional fingerprint array (n_molecules, n_features)
        smiles: Optional list of SMILES strings
        _validation_report: Input validation results
        _fp_type: Detected fingerprint type
    
    Example:
        >>> clusters = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
        >>> fps = np.random.randint(0, 2, (1000, 2048))
        >>> analyzer = ClusterAnalyzer(clusters=clusters, fps=fps)
        >>> stats = analyzer.get_basic_stats()
    """
    
    def __init__(
        self,
        clusters: List[List[int]],
        fps: Optional[np.ndarray] = None,
        smiles: Optional[List[str]] = None,
        centroids: Optional[List[np.ndarray]] = None,
    ):
        """Initialize ClusterAnalyzer with cluster information and optional fingerprints/SMILES.
        
        Args:
            clusters: List of lists, where each inner list contains
                     molecule indices belonging to that cluster.
                     Indices must be valid (0 <= idx < n_molecules).
            fps: Optional fingerprint array of shape (n_molecules, n_features).
                 Must be numpy array with dtype compatible with similarity calculations.
            smiles: Optional list of SMILES strings, one per molecule.
                    Length must match fps.shape[0] if provided.
        
        Raises:
            TypeError: If input types are incorrect.
        """
        self.fps = fps
        self.smiles = smiles
        self.clusters = clusters
        self.centroids = centroids
        # Validate inputs (fps can be None now)
        if self.fps is not None and self.smiles is not None:
            if len(self.smiles) != self.fps.shape[0]:
                raise ValueError(
                    f"Length of smiles ({len(self.smiles)}) must match number of molecules in fps ({self.fps.shape[0]})"
                )
            
        if self.centroids is not None:
            if len(self.centroids) != len(self.clusters):
                raise ValueError(
                    f"Length of centroids ({len(self.centroids)}) must match number of clusters ({len(self.clusters)})"
                )
    
    def cluster_population_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive population statistics for clustering results.

        Returns:
            Dictionary containing:
                - total_fps: Total number of fingerprints/molecules
                - total_clusters: Total number of clusters
                - total_singletons: Number of singleton clusters (size 1)
                - singleton_percentage: Percentage of clusters that are singletons
                - clusters_size_gt_10: Number of clusters with size > 10
                - clusters_size_gt_10_percentage: Percentage of clusters with size > 10
                - clusters_size_gt_100: Number of clusters with size > 100
                - clusters_size_gt_100_percentage: Percentage of clusters with size > 100
                - clusters_fps_ratio: Ratio of num_clusters to num_fps
                - mean_size: Mean cluster size
                - max_size: Maximum cluster size
                - q3_size: 75th percentile cluster size
                - median_size: Median cluster size
                - q1_size: 25th percentile cluster size
                - min_size: Minimum cluster size
        """
        cluster_sizes = np.array([len(cluster) for cluster in self.clusters])

        total_fps = np.sum(cluster_sizes)
        total_clusters = len(self.clusters)
        total_singletons = np.sum(cluster_sizes == 1)
        clusters_size_gt_10 = np.sum(cluster_sizes > 10)
        clusters_size_gt_100 = np.sum(cluster_sizes > 100)

        singleton_percentage = (total_singletons / total_clusters * 100) if total_clusters > 0 else 0
        clusters_gt_10_percentage = (clusters_size_gt_10 / total_clusters * 100) if total_clusters > 0 else 0
        clusters_gt_100_percentage = (clusters_size_gt_100 / total_clusters * 100) if total_clusters > 0 else 0
        clusters_fps_ratio = (total_clusters / total_fps) if total_fps > 0 else 0

        return {
            "total_fps": int(total_fps),
            "total_clusters": int(total_clusters),
            "total_singletons": int(total_singletons),
            "singleton_percentage": round(singleton_percentage, 2),
            "clusters_size_gt_10": int(clusters_size_gt_10),
            "clusters_size_gt_10_percentage": round(clusters_gt_10_percentage, 2),
            "clusters_size_gt_100": int(clusters_size_gt_100),
            "clusters_size_gt_100_percentage": round(clusters_gt_100_percentage, 2),
            "clusters_fps_ratio": round(clusters_fps_ratio, 2),
            "mean_size": round(float(np.mean(cluster_sizes)), 2),
            "max_size": int(np.max(cluster_sizes)),
            "q3_size": int(np.percentile(cluster_sizes, 75)),
            "median_size": int(np.median(cluster_sizes)),
            "q1_size": int(np.percentile(cluster_sizes, 25)),
            "min_size": int(np.min(cluster_sizes)),
        }
    

    def cluster_pop_distribution(self,
                                 top_clusters: int = 20,):
        """Calculate the populations of the top N largest clusters."""
        return np.array([len(cluster) for cluster in self.clusters[:top_clusters]])
    
    
    def populations_table(self,
                          top_clusters: int = 20,):
        """Generate a formatted table of cluster population statistics for the top N clusters."""
        cluster_sizes = self.cluster_pop_distribution(top_clusters=top_clusters)
        df = pd.DataFrame({
            "Cluster Index": list(range(top_clusters)),
            "Population": cluster_sizes
        })
        print(df.to_string(index=False, formatters={"Population": "{:,}".format}))
        print("\nOverall Population Statistics:")
        stats = self.cluster_population_stats()
        for key, value in stats.items():
            if isinstance(value, (int, np.integer)):
                print(f"{key}: {value:,}")
            else:
                print(f"{key}: {value}")


    def clusters_isim(self,
                      top_clusters: int = 20,):
        """Calculate the iSIM for each cluster."""
        cluster_isims = []
        if self.fps is None and self.smiles is None:
            raise ValueError("At least one of fps or smiles must be provided to calculate iSIM.")
        elif self.fps is None and self.smiles is not None:
            print("Recomputing fingerprints from smiles for iSIM calculation...")
            print("This may take some time for large clusters.")
            for cluster in self.clusters[:top_clusters]:
                fps_cluster = binary_fps([self.smiles[idx] for idx in cluster])
                cluster_isims.append(jt_isim(fps_cluster))
        else:
            for cluster in self.clusters[:top_clusters]:
                fps_cluster = self.fps[cluster]
                cluster_isims.append(jt_isim(fps_cluster))
        return np.array(cluster_isims)
    

    def isim_table(self,
                   top_clusters: int = 20,):
        """Generate a formatted table of iSIM values for the top N clusters."""
        cluster_isims = self.clusters_isim(top_clusters=top_clusters)
        print(pd.DataFrame({
            "Cluster Index": list(range(top_clusters)),
            "iSIM": np.round(cluster_isims, 4)
        }).to_string(index=False))


    def tanimoto_to_centroid(self,
                            top_clusters: int = 20,):
        """Calculate the Tanimoto similarity to the cluster centroid for each cluster."""
        if self.centroids is None:
            raise ValueError("Centroids must be provided to calculate Tanimoto to centroid.")
        if self.fps is None and self.smiles is not None:
            print("Recomputing fingerprints from smiles for Tanimoto to centroid calculation...")
            print("This may take some time for large clusters.")

        distributions = self._tanimoto_to_centroid_distributions(top_clusters=top_clusters)
        cluster_centroid_sims_stat = []
        for dist in distributions:
            cluster_centroid_sims_stat.append([
                np.min(dist),
                np.mean(dist),
                np.max(dist)])
        return np.array(cluster_centroid_sims_stat)


    def tanimoto_to_centroid_table(self,
                                  top_clusters: int = 20,):
        """Generate a formatted table of Tanimoto similarity to centroid for the top N clusters."""
        cluster_centroid_sims_stat = self.tanimoto_to_centroid(top_clusters=top_clusters)
        df = pd.DataFrame({
            "Min Tanimoto to Centroid": np.round(cluster_centroid_sims_stat[:, 0], 4),
            "Mean Tanimoto to Centroid": np.round(cluster_centroid_sims_stat[:, 1], 4),
            "Max Tanimoto to Centroid": np.round(cluster_centroid_sims_stat[:, 2], 4),
        })
        df.index.name = "Cluster Index"
        print(df.to_string())

    def _tanimoto_to_centroid_distributions(self,
                                            top_clusters: int = 20,):
        """Get the full distribution of Tanimoto similarities for each cluster to its centroid."""
        if self.centroids is None:
            raise ValueError("Centroids must be provided to calculate Tanimoto to centroid.")
        if self.fps is None and self.smiles is None:
            raise ValueError("At least one of fps or smiles must be provided to calculate Tanimoto to centroid.")

        distributions = []
        if self.fps is None and self.smiles is not None:
            for k, cluster in enumerate(self.clusters[:top_clusters]):
                fps_cluster = binary_fps([self.smiles[idx] for idx in cluster])
                centroids_sims = jt_sim_packed(self.centroids[k], fps_cluster)
                distributions.append(centroids_sims)
        else:
            for k, cluster in enumerate(self.clusters[:top_clusters]):
                fps_cluster = self.fps[cluster]
                centroids_sims = jt_sim_packed(self.centroids[k], fps_cluster)
                distributions.append(centroids_sims)
        return distributions

    def tanimoto_to_centroid_plot(self,
                                  top_clusters: int = 20,
                                  save_path: Optional[str] = None,
                                  figsize: tuple = (12, 6)):
        """Generate a violin plot showing Tanimoto similarity distributions to centroid for top clusters.

        Args:
            top_clusters (int): Number of top clusters to display. Defaults to 20.
            save_path (str, optional): Path to save the plot. Defaults to None.
            figsize (tuple): Figure size (width, height). Defaults to (12, 6).
        """
        distributions = self._tanimoto_to_centroid_distributions(top_clusters=top_clusters)

        plot_data = []
        cluster_labels = []
        for cluster_idx, dist in enumerate(distributions):
            plot_data.extend(dist)
            cluster_labels.extend([cluster_idx] * len(dist))

        df_plot = pd.DataFrame({
            "Cluster": cluster_labels,
            "Tanimoto Similarity": plot_data
        })

        fig, ax = plt.subplots(figsize=figsize)
        sns.violinplot(data=df_plot, x="Cluster", y="Tanimoto Similarity", ax=ax)

        ax.set_xlabel("Cluster Index")
        ax.set_ylabel("Tanimoto Similarity to Centroid")
        ax.set_title("Distribution of Tanimoto Similarities to Centroid")
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400)
        else:
            plt.show()

    def population_distribution_plot(self,
                                     save_path: Optional[str] = None,
                                     figsize: tuple = (6, 6)):
        """Plot the distribution of cluster sizes as a stacked bar chart.

        Shows how many clusters fall into size categories (>0, >1, >10, >100, >1000).

        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
            figsize (tuple): Figure size (width, height). Defaults to (6, 6).
        """
        cluster_sizes = np.array([len(cluster) for cluster in self.clusters])

        n_1000 = np.sum(cluster_sizes > 1000)
        n_100 = np.sum(cluster_sizes > 100)
        n_10 = np.sum(cluster_sizes > 10)
        n_1 = np.sum(cluster_sizes > 1)
        n_0 = np.sum(cluster_sizes > 0)

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar('Num_clusters', n_0, label='>0', color='blue')
        ax.bar('Num_clusters', n_1, label='>1', color='orange')
        ax.bar('Num_clusters', n_10, label='>10', color='gray')
        ax.bar('Num_clusters', n_100, label='>100', color='green')
        ax.bar('Num_clusters', n_1000, label='>1000', color='red')
        ax.legend()
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Cluster Size Distribution')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400)
        else:
            plt.show()

    def populations_isim_plot(self,
                              top_clusters: int = 20,
                              isim: bool = True,
                              save_path: Optional[str] = None,
                              figsize: tuple = (12, 6)):
        """Plot cluster populations with optional iSIM values overlay.

        Args:
            top_clusters (int): Number of top clusters to display. Defaults to 20.
            isim (bool): Whether to include iSIM values on secondary axis. Defaults to True.
            save_path (str, optional): Path to save the plot. Defaults to None.
            figsize (tuple): Figure size (width, height). Defaults to (12, 6).
        """
        populations = self.cluster_pop_distribution(top_clusters=top_clusters)

        fig, ax1 = plt.subplots(figsize=figsize)

        x = np.arange(len(populations))
        bars = ax1.bar(x, populations, alpha=0.7, color='blue', label='Population')
        ax1.set_xlabel('Cluster Index')
        ax1.set_ylabel('Population', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i) for i in x], rotation=45, ha='right')

        if isim:
            isim_values = self.clusters_isim(top_clusters=top_clusters)
            ax2 = ax1.twinx()
            line = ax2.plot(x, isim_values, color='darkorange', marker='o',
                           linewidth=2, markersize=6, label='iSIM')
            ax2.set_ylabel('iSIM', color='darkorange')
            ax2.tick_params(axis='y', labelcolor='darkorange')
            ax2.set_ylim(0, 1)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')

        stats = self.cluster_population_stats()
        annotation_text = f"Total Clusters: {stats['total_clusters']:,}\nSingletons: {stats['total_singletons']:,}"
        ax1.text(0.98, 0.98, annotation_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax1.set_title('Cluster Population' + (' and iSIM' if isim else ''))

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400)
        else:
            plt.show()

    def top_clusters_MCS_images(self,
                                top_clusters: int = 5,
                                mols_per_row: int = 3,
                                save_path: Optional[str] = None):
        """Generate grid images of Maximum Common Substructure (MCS) for top clusters.

        For each top cluster, displays the MCS with atoms highlighted in a grid layout.

        Args:
            top_clusters (int): Number of top clusters to display. Defaults to 5.
            mols_per_row (int): Number of MCS images per row in grid. Defaults to 3.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        if self.smiles is None:
            raise ValueError("SMILES data is required to generate MCS images.")

        n_rows = (top_clusters + mols_per_row - 1) // mols_per_row
        fig, axes = plt.subplots(n_rows, mols_per_row, figsize=(5 * mols_per_row, 5 * n_rows))
        if n_rows == 1 and mols_per_row == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif mols_per_row == 1:
            axes = [[ax] for ax in axes]

        axes_flat = [ax for row in axes for ax in row]

        for idx, cluster in enumerate(self.clusters[:top_clusters]):
            cluster_smiles = [self.smiles[i] for i in cluster]
            mcs_img = MSC_image(cluster_smiles,
                                standarize=True)
            axes_flat[idx].imshow(mcs_img)
            axes_flat[idx].set_title(f"Cluster {idx} - MCS (Size: {len(cluster)} molecules)")
            axes_flat[idx].axis('off')

        for idx in range(top_clusters, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches='tight')
        else:
            plt.show()

    def nearest_to_centroid_images(self,
                                   top_clusters: int = 5,
                                   mols_per_row: int = 5,
                                   sub_img_size: tuple = (250, 250),
                                   save_path: Optional[str] = None):
        """Display molecules nearest to centroid for top clusters in a grid.

        For each top cluster, finds the molecule with highest Tanimoto similarity
        to the centroid and displays them in a grid with similarity scores as legends.

        Args:
            top_clusters (int): Number of top clusters to display. Defaults to 5.
            mols_per_row (int): Number of molecules per row in grid. Defaults to 5.
            sub_img_size (tuple): Size of each sub-image. Defaults to (250, 250).
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        from rdkit.Chem import Draw #type: ignore

        if self.smiles is None:
            raise ValueError("SMILES data is required to display molecules.")
        if self.centroids is None:
            raise ValueError("Centroids are required to find nearest molecules.")

        nearest_smiles = []
        legends = []

        for idx, cluster in enumerate(self.clusters[:top_clusters]):
            if self.fps is None:
                fps_cluster = binary_fps([self.smiles[i] for i in cluster])
            else:
                fps_cluster = self.fps[cluster]

            centroid_sims = jt_sim_packed(self.centroids[idx], fps_cluster)
            nearest_idx = np.argmax(centroid_sims)
            mol_idx = cluster[nearest_idx]
            nearest_smiles.append(self.smiles[mol_idx])
            legends.append(f"C{idx}: {centroid_sims[nearest_idx]:.3f}")

        mols = [Chem.MolFromSmiles(smi) for smi in nearest_smiles]
        mols = [m for m in mols if m is not None]

        img = Draw.MolsToGridImage(mols,
                                   molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size,
                                   legends=legends[:len(mols)],
                                   useSVG=False)

        if save_path:
            img.save(save_path)
        else:
            img.show()