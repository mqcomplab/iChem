import numpy as np # type: ignore
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .libchem import LibChem

from ._libchem_aux import (
    interiSIM,
    intraiSIM,
    MaxSum,
    MinSum,
    combo_counts,
    composition_per_cluster,
)
from ..bblean.similarity import jt_sim_matrix_between_packed


class LibComparison:
    """Class for comparing multiple chemical libraries.
    
    This class provides functionality for comparing chemical libraries using various
    similarity metrics, clustering combined libraries, and visualizing comparison results.
    It works with LibChem instances and supports pairwise comparisons, medoid-based
    comparisons, and cluster composition analysis.
    
    Attributes:
        libraries (dict): Dictionary of LibChem instances keyed by library name.
        library_names (list): List of library names in order of addition.
        combined_library (LibChem): Combined LibChem instance created from clustering.
    """
    def __init__(self, 
    ):
        self.libraries: dict = {}
        self.library_names: list = []
        self.combined_library: 'LibChem | None' = None

    def add_library(self,
            library,
            lib_name: str,
    ) -> None:
        """Add a chemical library to the comparison.
        
        Registers a LibChem instance with a unique name for subsequent comparisons.
        
        Args:
            library (LibChem): The LibChem instance to add to the comparison.
            lib_name (str): Unique identifier for the library.
        """
        self.libraries[lib_name] = library
        self.library_names.append(lib_name)

    def compare_libraries(self,
                        methodology: str = 'intraiSIM',
                        lib1_name: str = None,
                        lib2_name: str = None) -> float:
        """Compare two libraries using full molecular fingerprints.
        
        Computes similarity between complete libraries using various metrics.
        If only two libraries exist and names are not specified, compares those two.
        
        Args:
            methodology (str, optional): Comparison methodology. Defaults to 'intraiSIM'.
                - 'intraiSIM': Intrinsic intra-library similarity comparison
                - 'interiSIM': Intrinsic inter-library similarity comparison
                - '_MaxSum': Maximum sum of pairwise similarities (hidden, computationally expensive)
                - '_MinSum': Minimum sum of pairwise similarities (hidden, computationally expensive)
            lib1_name (str, optional): Name of the first library. If None and only two libraries
                exist, automatically uses both. Defaults to None.
            lib2_name (str, optional): Name of the second library. If None and only two libraries
                exist, automatically uses both. Defaults to None.
        
        Returns:
            float: Similarity score between the two libraries.
            
        Raises:
            ValueError: If specified libraries don't exist or if library names are required but not provided.
        """
        if lib1_name is None and lib2_name is None and len(self.libraries) == 2:
            lib1_name, lib2_name = list(self.libraries.keys())
        if lib1_name not in self.library_names or lib2_name not in self.library_names:
            raise ValueError("Both libraries must be specified and exist in the comparison libraries.")
        

        lib1 = self.libraries[lib1_name]
        lib2 = self.libraries[lib2_name]

        fps1 = lib1.get_fingerprints(packed=True)
        fps2 = lib2.get_fingerprints(packed=True)

        if methodology == 'intraiSIM':
            isim_value = intraiSIM(
                np.array(fps1),
                np.array(fps2),
            )
            return isim_value
        if methodology == 'interiSIM':
            return interiSIM(
                np.array(fps1),
                np.array(fps2),
            )
        if methodology == '_MaxSum': # Note: this is hidden from user to prevent large number of pairwise calculations
            sim_matrix = jt_sim_matrix_between_packed(
                np.array(fps1),
                np.array(fps2),
            )
            return MaxSum(sim_matrix)
        if methodology == '_MinSum': # Note: this is hidden from user to prevent large number of pairwise calculations
            sim_matrix = jt_sim_matrix_between_packed(
                np.array(fps1),
                np.array(fps2),
            )
            return MinSum(sim_matrix)

    def compare_medoids(self, 
                        methodology: str = 'MaxSum',
                        lib1_name: str = None,
                        lib2_name: str = None,
                        ) -> float:
        """Compare cluster medoids between two libraries.
        
        Retrieves medoid molecules from each cluster and compares them using the specified
        methodology. This is more efficient than full library comparison.
        
        Args:
            methodology (str, optional): Comparison methodology. Defaults to 'MaxSum'.
                - 'MaxSum': Maximum sum of pairwise similarities
                - 'MinSum': Minimum sum of pairwise similarities
                - 'intraiSIM': Intrinsic intra-library similarity
                - 'interiSIM': Intrinsic inter-library similarity
            lib1_name (str, optional): Name of the first library. If None and only two libraries
                exist, automatically uses both. Defaults to None.
            lib2_name (str, optional): Name of the second library. If None and only two libraries
                exist, automatically uses both. Defaults to None.
        
        Returns:
            float: Similarity score between medoids of the two libraries.
            
        Raises:
            ValueError: If specified libraries don't exist or if library names are required but not provided.
            """
        if lib1_name is None and lib2_name is None and len(self.libraries) == 2:
            lib1_name, lib2_name = list(self.libraries.keys())
        if lib1_name not in self.library_names or lib2_name not in self.library_names:
            raise ValueError("Both libraries must be specified and exist in the comparison libraries.")
        
        
        lib1 = self.libraries[lib1_name]
        lib2 = self.libraries[lib2_name]

        fps1 = lib1.get_cluster_medoids(return_smiles=False)
        fps2 = lib2.get_cluster_medoids(return_smiles=False)

        if methodology == 'MaxSum':
            sim_matrix = jt_sim_matrix_between_packed(
            np.array(fps1),
            np.array(fps2),
        )
            return MaxSum(sim_matrix)
        
        if methodology == 'MinSum':
            sim_matrix = jt_sim_matrix_between_packed(
            np.array(fps1),
            np.array(fps2),
        )
            return MinSum(sim_matrix)

        if methodology == 'intraiSIM':
            return intraiSIM(
                np.array(fps1),
                np.array(fps2),
            )
        
        if methodology == 'interiSIM':
            return interiSIM(
                np.array(fps1),
                np.array(fps2),
            )
        
    def compare_medoids_all(self,
                            methodology: str = 'MaxSum',
                            ) -> list[list[float]]:
        """Compare medoids between all library pairs.
        
        Performs pairwise medoid comparisons for all libraries and returns a symmetric
        similarity matrix. Uses symmetry to avoid redundant calculations.
        
        Args:
            methodology (str, optional): Comparison methodology. Defaults to 'MaxSum'.
                - 'MaxSum': Maximum sum of pairwise similarities
                - 'MinSum': Minimum sum of pairwise similarities
                - 'intraiSIM': Intrinsic intra-library similarity
                - 'interiSIM': Intrinsic inter-library similarity
        
        Returns:
            list[list[float]]: Symmetric similarity matrix where element [i][j] represents
                the similarity between library i and library j.
        """
        results = []

        for i, library_1 in enumerate(self.library_names):
            result_line = []
            for j, library_2 in enumerate(self.library_names):
                if j < i:  # Use the symmetrical value
                    result_line.append(results[j][i])
                else:  # Compute the value
                    k = self.compare_medoids(methodology=methodology,
                                            lib1_name=library_1,
                                            lib2_name=library_2)
                    result_line.append(k)
            results.append(result_line)

        return results

    def compare_medoids_heatmap(self,
                                methodology: str = 'MaxSum',
                                save_path: str = None) -> None:
        """Generate a heatmap visualizing medoid comparisons across all libraries.
        
        Creates a symmetric heatmap showing pairwise similarities between all libraries
        based on their cluster medoids.
        
        Args:
            methodology (str, optional): Comparison methodology. Defaults to 'MaxSum'.
                - 'MaxSum': Maximum sum of pairwise similarities
                - 'MinSum': Minimum sum of pairwise similarities
                - 'intraiSIM': Intrinsic intra-library similarity
                - 'interiSIM': Intrinsic inter-library similarity
            save_path (str, optional): File path to save the heatmap. If None, displays 
                the heatmap without saving. Defaults to None.
        """
        from ..visualization.plots import symmetric_heatmap
        results = self.compare_medoids_all(methodology=methodology)
        symmetric_heatmap(results,
                          labels=self.library_names,
                          save_path=save_path)
        
    def cluster_libraries(
            self,
            methodology: str = 'medoids',
            lib_names: list[str] = None,
            threshold: float = None,
            n_samples: int = None,
            verbose: bool = False,
    ) -> None:
        """Cluster molecules from multiple libraries into a combined library.
        
        Creates a combined library by merging molecules from selected libraries and
        performing hierarchical clustering. The combined library is stored in the
        combined_library attribute.
        
        Args:
            methodology (str, optional): Clustering approach. Defaults to 'medoids'.
                - 'medoids': Uses cluster medoids from each library
                - 'samples': Uses stratified samples from each library
            lib_names (list[str], optional): Libraries to include. If None, uses all 
                registered libraries. Defaults to None.
            threshold (float, optional): Custom clustering threshold. If None, uses the 
                maximum threshold among included libraries. Defaults to None.
            n_samples (int, optional): Number of samples per library when methodology is 
                'samples'. Ignored for 'medoids'. Defaults to None.
            verbose (bool, optional): If True, prints progress information. Defaults to False.
            
        Raises:
            ValueError: If methodology is not 'medoids' or 'samples'.
        """
        if methodology == 'medoids':
            combined_lib = self._cluster_medoid_mix(threshold=threshold,
                                                          lib_names=lib_names,
                                                          verbose=verbose)
        elif methodology == 'samples':
            combined_lib = self._cluster_sample_mix(n_samples=n_samples,
                                                         threshold=threshold,
                                                         lib_names=lib_names,
                                                         verbose=verbose)
        else:
            raise ValueError("Methodology must be either 'medoids' or 'samples'.")

        self.combined_library = combined_lib

    def _cluster_medoid_mix(self,
                            threshold: float = None,
                            lib_names: list[str] = None,
                            verbose: bool = False) -> 'LibChem':
        """Cluster combined medoids from multiple libraries.
        
        Internal method that creates a combined library from cluster medoids of selected
        libraries, shuffles them for unbiased clustering, and performs hierarchical clustering.
        
        Args:
            threshold (float, optional): Custom clustering threshold. If None, uses the 
                maximum threshold among included libraries. Defaults to None.
            lib_names (list[str], optional): Libraries to include. If None, uses all 
                registered libraries. Defaults to None.
            verbose (bool, optional): If True, prints progress information. Defaults to False.
        
        Returns:
            LibChem: Combined library with clustered medoids.
            
        Raises:
            ValueError: If fewer than two libraries are specified or if a library name is not found.
            
        Note:
            This is an internal method. Use cluster_libraries() instead.
        """
        if lib_names is None:
            lib_names = list(self.library_names)
        if len(lib_names) < 2:
            raise ValueError("At least two libraries must be specified for comparison.")
        
        medoids_fps = []
        medoids_smiles = []
        medoids_flags = []
        for lib_name in lib_names:
            if lib_name not in self.library_names:
                raise ValueError(f"Library '{lib_name}' not found in comparison libraries.")
            
            lib_medoids_fps, lib_medoids_smiles = self.libraries[lib_name].get_cluster_medoids(return_smiles=True)
            medoids_fps.extend(lib_medoids_fps)
            medoids_smiles.extend(lib_medoids_smiles)
            medoids_flags.extend([lib_name] * len(lib_medoids_smiles))
            if verbose:
                print(f'Number of medoids in Library {lib_name}: {len(lib_medoids_smiles)}')
            
        medoids_fps = np.array(medoids_fps)
        n_medoids = len(medoids_fps)
        
        if verbose:
            print(f'Total number of medoids: {n_medoids}')
        
        # Create a new LibChem instance for combined medoids
        from .libchem import LibChem
        combined_lib = LibChem()
        
        # Shuffle the fingerprints before clustering
        indices = np.arange(len(medoids_fps))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        medoids_fps = medoids_fps[indices]
        medoids_flags = [medoids_flags[i] for i in indices]
        medoids_smiles = [medoids_smiles[i] for i in indices]

        # Load the combined medoids into the new library
        combined_lib.set_fingerprints(
            fingerprints = medoids_fps,
            packed = True,
        )

        # Set the smiles and the flags
        combined_lib.set_smiles(medoids_smiles)
        combined_lib.set_flags(medoids_flags)

        # Define what threshold to use
        if threshold is None:
            combined_lib.set_threshold()
            threshold_combined = combined_lib.threshold
            library_thresholds = [self.libraries[name].threshold for name in lib_names if self.libraries[name].threshold is not None]
            library_thresholds.append(threshold_combined)
            threshold_final = max(library_thresholds)
            if verbose:
                print(f'Using clustering threshold: {threshold_final:.4f}')
        else:
            threshold_final = threshold
            if verbose:
                print(f'Using clustering threshold: {threshold_final:.4f}')

        # Do the clustering
        combined_lib.set_threshold(threshold=threshold_final)
        combined_lib.cluster()

        return combined_lib
    
    def _cluster_sample_mix(self,
                            n_samples: int = None,
                            threshold: float = None,
                            lib_names: list[str] = None,
                            verbose: bool = False) -> 'LibChem':
        """Cluster combined stratified samples from multiple libraries.
        
        Internal method that draws stratified samples from selected libraries, combines
        and shuffles them, then performs hierarchical clustering.
        
        Args:
            n_samples (int, optional): Number of samples to draw from each library. 
                Defaults to None.
            threshold (float, optional): Custom clustering threshold. If None, uses the 
                maximum threshold among included libraries. Defaults to None.
            lib_names (list[str], optional): Libraries to include. If None, uses all 
                registered libraries. Defaults to None.
            verbose (bool, optional): If True, prints progress information. Defaults to False.
        
        Returns:
            LibChem: Combined library with clustered samples.
            
        Raises:
            ValueError: If fewer than two libraries are specified or if a library name is not found.
            
        Note:
            This is an internal method. Use cluster_libraries() instead.
        """
        if lib_names is None:
            lib_names = list(self.library_names)
        if len(lib_names) < 2:
            raise ValueError("At least two libraries must be specified for comparison.")

        sampled_fps = []
        sampled_smiles = []
        sampled_flags = []

        # collect samples from each library
        for lib_name in lib_names:
            if lib_name not in self.library_names:
                raise ValueError(f"Library '{lib_name}' not found in comparison libraries.")
            lib = self.libraries[lib_name]
            # get_cluster_samples returns (fps, smiles) when return_smiles=True
            fps, smiles = lib.get_cluster_samples(n_samples=n_samples, return_smiles=True, return_cluster_ids=False)
            if verbose:
                print(f'Number of sampled mols in Library {lib_name}: {len(fps)}')
            sampled_fps.extend(list(fps))
            sampled_smiles.extend(smiles)
            sampled_flags.extend([lib_name] * len(fps))

        # combine and shuffle
        sampled_fps = np.array(sampled_fps)
        n_total = len(sampled_fps)
        if verbose:
            print(f'Total number of sampled mols: {n_total}')
        indices = np.arange(n_total)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        sampled_fps = sampled_fps[indices]
        sampled_smiles = [sampled_smiles[i] for i in indices]
        sampled_flags = [sampled_flags[i] for i in indices]

        # create combined LibChem and load fingerprints
        from .libchem import LibChem
        combined_lib = LibChem()
        combined_lib.set_fingerprints(fingerprints=sampled_fps, packed=True)
        
        # set smiles and flags on combined lib
        combined_lib.set_smiles(sampled_smiles)
        combined_lib.set_flags(sampled_flags)

        # determine threshold to use
        if threshold is None:
            combined_lib.set_threshold()
            threshold_combined = combined_lib.threshold
            library_thresholds = [self.libraries[name].threshold for name in lib_names if self.libraries[name].threshold is not None]
            library_thresholds.append(threshold_combined)
            threshold_final = max(library_thresholds)
            if verbose:
                print(f'Using clustering threshold: {threshold_final:.4f}')
        else:
            threshold_final = threshold
            if verbose:
                print(f'Using clustering threshold: {threshold_final:.4f}')

        # cluster with the chosen threshold
        combined_lib.set_threshold(threshold=threshold_final)
        combined_lib.cluster()

        return combined_lib
    
    def cluster_classification_counts(self,
                            lib_names: list[str] = None,
                            ) -> tuple:
        """Get counts and mapping of cluster compositions by library.
        
        Analyzes the combined library clusters to determine which libraries contribute
        molecules to each cluster and in what proportions.
        
        Args:
            lib_names (list[str], optional): Libraries to include in analysis. If None, 
                uses all registered libraries. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - counts (dict): Dictionary mapping cluster composition patterns to counts
                - combo_map (dict): Dictionary mapping patterns to cluster indices
                
        Raises:
            ValueError: If no combined library exists (cluster_libraries() must be called first).
        """
        if self.combined_library is None:
            raise ValueError("No combined library found. Please run cluster_libraries() first.")
        
        # Use combo_counts helper to get exact mapping and counts
        lib_names = lib_names if lib_names is not None else list(self.library_names)
        counts, combo_map = combo_counts(self.combined_library.get_cluster_flags(), library_names=lib_names)
        return counts, combo_map
    
    def pie_chart_composition(self,
                              lib_names: list[str] = None,
                              save_path: str = None,
                              ) -> None:
        """Generate a pie chart showing cluster composition by library.
        
        Visualizes how molecules from different libraries are distributed across
        mixed clusters in the combined library.
        
        Args:
            lib_names (list[str], optional): Libraries to include in the chart, in the same 
                order used for clustering. If None, uses all libraries. Defaults to None.
            save_path (str, optional): File path to save the chart. If None, displays 
                without saving. Defaults to None.
                
        Raises:
            ValueError: If no combined library exists (cluster_libraries() must be called first).
        """
        from ..visualization.plots import pie_chart_mixed_clusters
        if self.combined_library is None:
            raise ValueError("No combined library found. Please run cluster_libraries() first.")
        labels = lib_names if lib_names is not None else list(self.library_names)
        counts, _ = combo_counts(self.combined_library.get_cluster_flags(), library_names=labels)
        pie_chart_mixed_clusters(counts, save_path=save_path)

    def venn_diagram_composition(self,
                                lib_names: list[str] = None,
                                save_path: str = None,
                                upset: bool = False,
                                ) -> None:
        """Generate a Venn diagram showing overlap in cluster compositions.
        
        Visualizes the overlap and unique contributions of different libraries to
        the combined cluster structure using a Venn diagram.

        Args:
            lib_names (list[str], optional): Libraries to include in the diagram, in the same 
                order used for clustering. If None, uses all libraries. Defaults to None.
            save_path (str, optional): File path to save the diagram. If None, displays 
                without saving. Defaults to None.
            upset (bool, optional): If True, generate an UpSet plot instead of a Venn diagram. 
                Defaults to False.
                
        Raises:
            ValueError: If no combined library exists (cluster_libraries() must be called first).
        """
        from ..visualization.plots import venn_lib_comp
        if self.combined_library is None:
            raise ValueError("No combined library found. Please run cluster_libraries() first.")
        counts, _ = combo_counts(self.combined_library.get_cluster_flags(), library_names=lib_names if lib_names is not None else self.library_names)
        venn_lib_comp(counts, lib_names=lib_names if lib_names is not None else self.library_names, save_path=save_path, upset=upset)

    def cluster_composition_counts(self,
                                   top: int = 20
                                   ) -> list[Counter]:
        """Get detailed composition of the largest clusters.
        
        Analyzes the top N largest clusters to determine which libraries contribute
        molecules to each cluster.
        
        Args:
            top (int, optional): Number of largest clusters to analyze. Defaults to 20.
            
        Returns:
            list[Counter]: List of Counter objects, one per cluster, mapping library 
                names to molecule counts in that cluster.
                
        Raises:
            ValueError: If no combined library exists (cluster_libraries() must be called first).
        """
        if self.combined_library is None:
            raise ValueError("No combined library found. Please run cluster_libraries() first.")
        composition = composition_per_cluster(
            self.combined_library.get_cluster_flags(),
            top=top,
        )
        return composition
    
    def plot_cluster_composition(self,
                                lib_names: list[str],
                                top: int = 20,
                                save_path: str = None,
                                 ) -> None:
        """Generate a bar chart showing composition of the largest clusters.
        
        Creates a stacked bar chart displaying how molecules from different libraries
        are distributed across the top N largest clusters.
        
        Args:
            lib_names (list[str]): Libraries to include in the chart, in the same order 
                used for clustering. This is a required parameter.
            top (int, optional): Number of largest clusters to plot. Defaults to 20.
            save_path (str, optional): File path to save the chart. If None, displays 
                without saving. Defaults to None.
                
        Raises:
            ValueError: If no combined library exists (cluster_libraries() must be called first).
        """
        from ..visualization.plots import bar_chart_library_comparison
        if self.combined_library is None:
            raise ValueError("No combined library found. Please run cluster_libraries() first.")
        composition = composition_per_cluster(
            self.combined_library.get_cluster_flags(),
            top=top,
        )
        bar_chart_library_comparison(
            values = composition,
            lib_names = lib_names,
            save_path = save_path,
        )

    def cluster_visualization(self,
                              cluster_number: int,
                              save_path: str = None,
                              ):
        """Visualize molecules in a specific cluster with Maximum Common Substructure (MCS).
        
        Generates a grid image showing representative molecules from the specified cluster,
        highlighting the maximum common substructure and color-coding by library origin.
        
        Args:
            cluster_number (int): Zero-based index of the cluster to visualize.
            save_path (str, optional): File path to save the image. If None, returns the 
                image object without saving. Defaults to None.
        
        Returns:
            PIL.Image or None: Returns the image object if save_path is None, otherwise None.
            
        Raises:
            ValueError: If no combined library exists (cluster_libraries() must be called first).
        """
        if self.combined_library is None:
            raise ValueError("No combined library found. Please run cluster_libraries() first.")
        from ..visualization.mol_images import cluster_mix_MCS_image
        if cluster_number > len(self.combined_library.clusters) - 1:
            raise ValueError(f"Cluster number {cluster_number} is out of range. There are only {len(self.combined_library.clusters)} clusters.")
        img = cluster_mix_MCS_image(
            cluster = self.combined_library.clusters[cluster_number],
            smiles = self.combined_library.smiles,
            flags = self.combined_library.flags,
            n_samples = 25,
            MCS_threshold = 0.75,
            save_path = save_path,
        )
        if save_path is None:
            return img
