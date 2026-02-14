import numpy as np # type: ignore
from collections import Counter

from ._libchem_aux import (
    interiSIM,
    intraiSIM,
    MaxSum,
    MinSum,
    combo_counts,
    composition_per_cluster,
)
from ..bblean.similarity import jt_sim_matrix_between_packed, jt_stratified_sampling


class LibComparison:
    """Class for comparing multiple chemical libraries. Uses libchem class instances to compare them."""
    def __init__(self, 
    ):
        self.libraries: dict = {}
        self.library_names: list = []
        self.combined_library: object = None

    def add_library(self,
            library,
            lib_name: str,
    ) -> None:
        """Add a library to the comparison
        
        Args:
            library (LibChem): The library instance to add.
            lib_name (str): The name to assign to the library.
        """
        self.libraries[lib_name] = library
        self.library_names.append(lib_name)

    def compare_libraries(self,
                        methodology: str = 'intraiSIM',
                        lib1_name: str = None,
                        lib2_name: str = None) -> float:
        """Compare two libraries using specified methodology. Uses the full fingerprints of the libraries.
        
        Args:
            methodology (str): The comparison methodology to use. Options are 'intraiSIM', 'interiSIM', '_MaxSum', '_MinSum'.
            lib1_name (str): The name of the first library to compare. If None and only two libraries are present, both will be used.
            lib2_name (str): The name of the second library to compare. If None and only two libraries are present, both will be used.
        
        Returns:
            float: The similarity score between the two libraries.
        """
        if lib1_name == None and lib2_name == None and len(self.libraries) == 2:
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
        """Compare medoids of two libraries, medoids from each cluster are retrieved and compared.
        
        Args:
            methodology (str): The comparison methodology to use. Options are 'MaxSum', 'MinSum', 'intraiSIM', 'interiSIM'.
            lib1_name (str): The name of the first library to compare. If None and only two libraries are present, both will be used.
            lib2_name (str): The name of the second library to compare. If None and only two libraries are present, both will be used.
            """
        if lib1_name == None and lib2_name == None and len(self.libraries) == 2:
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
        """Compare medoids of all possible permutations of library pairs and return a results dictionary.
        
        Args:
            methodology (str): The comparison methodology to use. Options are 'MaxSum', 'MinSum', 'intraiSIM', 'interiSIM'.
        
        Returns:
            list[list[float]]: A list of lists with pairwise comparison results.
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
        """Generate a heatmap of medoid comparisons across all libraries.
        Args:
            methodology (str): The comparison methodology to use. Options are 'MaxSum', 'MinSum', 'intraiSIM', 'interiSIM'.
            save_path (str): Path to save the heatmap image. If None, the heatmap is displayed but not saved.
            stratified_samples (int): Number of stratified samples for heatmap generation. Defaults to 50."""
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
            factor: float = 3.5,
            n_samples: int = None,
            verbose: bool = False,
    ) -> None:
        """Cluster the combined libraries using specified methodology. Indicate which libraries to compare, otherwise all are used.
        
        Args:
            methodology (str): The clustering methodology to use. Options are 'medoids' or 'samples'.
            lib_names (list[str]): List of library names to include in clustering. If None, all libraries are used.
            threshold (float): Custom threshold value for clustering. If None, uses the maximum threshold among libraries.
            factor (float): Factor to adjust the clustering threshold in terms of standard deviations. Defaults to 3.5.
            n_samples (int): Number of samples to draw from each library when using 'samples' methodology. Ignored if 'medoids' is used.
            verbose (bool): Whether to print progress messages.
        """
        if methodology == 'medoids':
            combined_lib = self._cluster_medoid_mix(threshold=threshold,
                                                          lib_names=lib_names,
                                                          factor=factor,
                                                          verbose=verbose)
        elif methodology == 'samples':
            combined_lib = self._cluster_sample_mix(n_samples=n_samples,
                                                         threshold=threshold,
                                                         factor=factor,
                                                         lib_names=lib_names,
                                                         verbose=verbose)
        else:
            raise ValueError("Methodology must be either 'medoids' or 'samples'.")

        self.combined_library = combined_lib

    def _cluster_medoid_mix(self,
                            threshold: float = None,
                            lib_names: list[str] = None,
                            factor: float = 3.5,
                            verbose: bool = False) -> object:
        """Cluster the combined medoids of both libraries.
        
        Args:
            threshold (float): Custom threshold value for clustering. If None, uses the maximum threshold among libraries.
            lib_names (list[str]): List of library names to include in clustering. If None, all libraries are used.
            verbose (bool): Whether to print progress messages.
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
            lib_medoids_fps, lib_medoids_smiles = self.libraries[lib_name].get_cluster_medoids(return_smiles=True,
                                                                                               threshold=threshold,
                                                                                               factor=factor)
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
        np.random.shuffle(indices)
        medoids_fps = medoids_fps[indices]
        medoids_flags = [medoids_flags[i] for i in indices]
        medoids_smiles = [medoids_smiles[i] for i in indices]

        # Load the combined medoids into the new library
        combined_lib.load_fingerprints(
            fingerprints = medoids_fps,
            packed = True,
        )

        # Set the smiles and the flags
        combined_lib.smiles = medoids_smiles
        combined_lib.load_flags(medoids_flags)

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
                            factor: float = 3.5,
                            lib_names: list[str] = None,
                            verbose: bool = False):
        """Cluster a sample (with stratified sampling) of combined samples of both libraries. 
        
        Args:
            n_samples (int): Number of samples to draw from each library.
            threshold (float): Custom threshold value for clustering. If None, uses the maximum threshold among libraries.
            factor (float): Factor to adjust the clustering threshold in terms of standard deviations. Defaults to 3.5.
            lib_names (list[str]): List of library names to include in clustering. If None, all libraries are used.
            verbose (bool): Whether to print progress messages.
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
            fps, smiles = lib.get_cluster_samples(n_samples=n_samples, return_smiles=True, threshold=threshold, factor=factor)
            fps = np.array(fps)
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
        np.random.shuffle(indices)
        sampled_fps = sampled_fps[indices]
        sampled_smiles = [sampled_smiles[i] for i in indices]
        sampled_flags = [sampled_flags[i] for i in indices]

        # create combined LibChem and load fingerprints
        from .libchem import LibChem
        combined_lib = LibChem()
        combined_lib.load_fingerprints(fingerprints=sampled_fps, packed=True)
        
        # set smiles and flags on combined lib
        combined_lib.smiles = sampled_smiles
        combined_lib.load_flags(sampled_flags)

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
                            threshold: float = None,
                            n_samples: int = None,
                            methodology: str = 'medoids',
                            verbose: bool = False,
                            factor: float = 3.5,
                            ) -> dict:
        """Cluster the combined medoids of libraries and return counts mapping.
        Args:
            lib_names (list[str]): List of library names to include in clustering. If None, all libraries are used.
            threshold (float): Custom threshold value for clustering. If None, uses the maximum threshold among libraries.
            n_samples (int): Number of samples to draw from each library when using 'samples' methodology. Ignored if 'medoids' is used.
            methodology (str): The clustering methodology to use. Options are 'medoids' or 'samples'.
            verbose (bool): Whether to print progress messages.
            factor (float): Factor to adjust the clustering threshold in terms of standard deviations. Defaults to 3.5.

        Returns:
            dict: A dictionary with counts of cluster classifications across libraries.
        """
        self.cluster_libraries(methodology=methodology,
                                   lib_names=lib_names,
                                   threshold=threshold,
                                   n_samples=n_samples,
                                   verbose=verbose, 
                                   factor=factor)
        # Use combo_counts helper to get exact mapping and counts
        lib_names = lib_names if lib_names is not None else list(self.library_names)
        counts, combo_map = combo_counts(self.combined_library.get_cluster_flags(), library_names=lib_names)
        return counts, combo_map
    
    def pie_chart_composition(self,
                              lib_names: list[str] = None,
                              save_path: str = None,
                              ) -> None:
        """Generate a pie chart of the combined library cluster compositions.
        
        Args:
            lib_names (list[str]): List of library names to include in the chart, must be in the same order as used in clustering. If None, all libraries are used.
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
                                ) -> None:
        """Generate a Venn diagram of the combined library cluster compositions.

        Args:
            lib_names (list[str]): List of library names to include in the diagram, must be in the same order as used in clustering. If None, all libraries are used.
            save_path (str): Path to save the Venn diagram image. If None, the diagram is displayed but not saved.

        Returns:
            None: Displays or saves the Venn diagram image.
        """
        from ..visualization.plots import venn_lib_comp
        if self.combined_library is None:
            raise ValueError("No combined library found. Please run cluster_libraries() first.")
        counts, _ = combo_counts(self.combined_library.get_cluster_flags(), library_names=lib_names if lib_names is not None else self.library_names)
        venn_lib_comp(counts, lib_names=lib_names if lib_names is not None else self.library_names, save_path=save_path)

    def cluster_composition_counts(self,
                                   top: int = 20
                                   ) -> list[Counter]:
        """Get the composition of the top clusters in the combined library.
        
        Args:
            top (int): Number of top clusters to retrieve composition for.
            
        Returns:
            list[Counter]: A list of Counters representing the composition of each top cluster.
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
                                 ):
        """Plot the composition of the top clusters in the combined library.
        
        Args:
            lib_names (list[str]): List of library names to include in the chart, must be in the same order as used in clustering.
            top (int): Number of top clusters to plot composition for.
            save_path (str): Path to save the bar chart image. If None, the chart is displayed but not saved.
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
                              ) -> object:
        """Generate and return/display/save cluster visualization of combined library.
        
        Args:
            cluster_number (int): The cluster number to visualize.
            save_path (str): Path to save the cluster image. If None, the image is returned but not saved.
        """
        if self.combined_library is None:
            raise ValueError("No combined library found. Please run cluster_libraries() first.")
        from ..visualization.mol_images import cluster_mix_MCS_image
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
