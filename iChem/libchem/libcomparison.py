import numpy as np # type: ignore

from ._libchem_aux import (
    interiSIM,
    intraiSIM,
    MaxSum,
    MinSum,
    combo_counts,
    cluster_class_counts,
    cluster_classification,
    calc_proportions,
)
from ..bblean.similarity import jt_sim_matrix_between_packed


class LibComparison:
    def __init__(self, 
    ):
        self.libraries = {}
        self.library_names = []
        self.combined_lib = None

    def add_library(self,
            library,
            lib_name: str,
    ) -> None:
        """Add a library to the comparison"""
        self.libraries[lib_name] = library
        self.library_names.append(lib_name)

    def compare_libraries(self,
                        methodology: str = 'intraiSIM',
                        lib1_name: str = None,
                        lib2_name: str = None) -> float:
        """Compare two libraries using specified methodology"""
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
                        lib2_name: str = None) -> float:
        """Compare medoids of two libraries"""
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
        
    def cluster_libraries(
            self,
            methodology: str = 'medoids',
            lib_names: list[str] = None,
            threshold: float = None,
            n_samples: int = None,
            verbose: bool = False,
    ) -> None:
        """Cluster the combined libraries using specified methodology"""
        if methodology == 'medoids':
            self.combined_lib = self._cluster_medoid_mix(threshold=threshold,
                                                          lib_names=lib_names,
                                                          verbose=verbose)
        if methodology == 'samples':
            self.combined_lib = self._cluster_sample_mix(n_samples=n_samples,
                                                         threshold=threshold,
                                                         lib_names=lib_names,
                                                         verbose=verbose)
        else:
            raise ValueError("Methodology must be either 'medoids' or 'samples'.")


    def _cluster_medoid_mix(self,
                            threshold: float = None,
                            lib_names: list[str] = None,
                            verbose: bool = False) -> object:
        """Cluster the combined medoids of both libraries"""
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
        np.random.shuffle(indices)
        medoids_fps = medoids_fps[indices]
        medoids_flags = [medoids_flags[i] for i in indices]
        medoids_smiles = [medoids_smiles[i] for i in indices]

        # Load the combined medoids into the new library
        combined_lib.load_fingerprints(
            fingerprints = medoids_fps,
            packed = True,
        )

        # Define what threshold to use
        if threshold is None:
            combined_lib.set_threshold()
            threshold_combined = combined_lib.threshold

        # Check which threshold is the best to use
        library_thresholds = [self.libraries[lib_name].threshold for lib_name in lib_names if self.libraries[lib_name].threshold is not None]
        library_thresholds.append(threshold_combined)

        threshold = max(library_thresholds)

        # Do the clustering
        combined_lib.set_threshold(threshold=threshold)
        combined_lib.cluster()

        # Set the smiles and the flags
        combined_lib.smiles = medoids_smiles
        combined_lib.load_flags(medoids_flags)

        return combined_lib
    
    def _cluster_sample_mix(self,
                            n_samples: int = None,
                            threshold: float = None,
                            lib_names: list[str] = None,
                            verbose: bool = False):
        """Cluster a sample of combined samples of both libraries"""
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
            fps, smiles = lib.get_cluster_samples(n_samples=n_samples, return_smiles=True)
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

        # determine threshold to use
        if threshold is None:
            combined_lib.set_threshold()
            threshold_combined = combined_lib.threshold
        else:
            threshold_combined = threshold

        library_thresholds = [self.libraries[name].threshold for name in lib_names if self.libraries[name].threshold is not None]
        library_thresholds.append(threshold_combined)
        threshold_final = max(library_thresholds)

        # cluster with the chosen threshold
        combined_lib.set_threshold(threshold=threshold_final)
        combined_lib.cluster()

        # set smiles and flags on combined lib
        combined_lib.smiles = sampled_smiles
        combined_lib.load_flags(sampled_flags)

        return combined_lib
    
    def cluster_classification_counts(self,
                            lib_names: list[str] = None,
                            threshold: float = None,
                            n_samples: int = None,
                            methodology: str = 'medoids',
                            ) -> dict:
        """Cluster the combined medoids of libraries and return counts mapping"""
        if self.combined_lib is None:
            self.cluster_libraries(methodology=methodology,
                                   lib_names=lib_names,
                                   threshold=threshold,
                                   n_samples=n_samples)
        combined_lib = self.combined_lib
        # Use combo_counts helper to get exact mapping and counts
        counts, combo_map = combo_counts(combined_lib.flags, library_names=list(self.library_names))
        return counts
    
    def get_mixed_cluster_indices(self,
                                  methodology: str = 'medoids',
                                  lib_names: list[str] = None,) -> dict:
        """Return mapping of combination keys -> cluster indices for mixed clusters"""
        if self.combined_lib is None:
            self.cluster_libraries(methodology=methodology, lib_names=lib_names)
        combined_lib = self.combined_lib
        _, combo_map = combo_counts(combined_lib.flags, library_names=list(self.library_names))
        # Filter only mixed (keys with '+')
        return {k: v for k, v in combo_map.items() if '+' in k}

    def cluster_sample_proportions(self,
                           n_samples: int = None,) -> float:
        """Cluster a sample of combined medoids of libraries and compute proportions"""
        if n_samples is None:
            raise ValueError("Number of samples must be specified.")

        # Get the clusters (combined LibChem) and compute counts
        combined_lib = self._cluster_sample_mix(n_samples=n_samples)
        # combined_lib is a LibChem instance
        clusters = combined_lib.get_clusters()
        n1 = 0  # unknown here, user should use cluster_class_counts directly if needed
        mixed_cluster_count, only_lib1_count, only_lib2_count = cluster_class_counts(clusters, n1=n1)
        proportion_mixed, proportion_1, proportion_2 = calc_proportions(mixed_cluster_count, only_lib1_count, only_lib2_count)
        return proportion_mixed, proportion_1, proportion_2

    def get_combined_lib_clusters(self,
                                  methodology: str = 'samples',
                                  n_samples: int = None,
                                  return_type: str = 'all',) -> list[list[int]]:
        """Get clusters from combined libraries based on specified methodology"""
        if methodology == 'medoids':
            combined_lib = self._cluster_medoid_mix()
        elif methodology == 'samples':
            if n_samples is None:
                raise ValueError("Number of samples must be specified for 'samples' methodology.")
            combined_lib = self._cluster_sample_mix(n_samples=n_samples)
        else:
            raise ValueError("Methodology must be either 'medoids' or 'samples'.")
        
        # combined_lib should be a LibChem instance with smiles and flags
        combined_smiles = []
        if hasattr(combined_lib, 'smiles') and combined_lib.smiles is not None:
            combined_smiles = combined_lib.smiles
        # Use cluster_classification to split clusters
        clusters = combined_lib.get_clusters()
        mixed, only_lib1, only_lib2 = cluster_classification(clusters, n1=0)

        if return_type == 'all':
            mixed_cluster_smiles = [[combined_smiles[idx] for idx in cluster] for cluster in mixed]
            only_lib1_cluster_smiles = [[combined_smiles[idx] for idx in cluster] for cluster in only_lib1]
            only_lib2_cluster_smiles = [[combined_smiles[idx] for idx in cluster] for cluster in only_lib2]
            return mixed_cluster_smiles, only_lib1_cluster_smiles, only_lib2_cluster_smiles
        elif return_type == 'mixed':
            mixed_cluster_smiles = [[combined_smiles[idx] for idx in cluster] for cluster in mixed]
            return mixed_cluster_smiles
        elif return_type == 'only_lib1':
            only_lib1_cluster_smiles = [[combined_smiles[idx] for idx in cluster] for cluster in only_lib1]
            return only_lib1_cluster_smiles
        elif return_type == 'only_lib2':
            only_lib2_cluster_smiles = [[combined_smiles[idx] for idx in cluster] for cluster in only_lib2]
            return only_lib2_cluster_smiles
        elif return_type == 'both_libs':
            only_lib1_cluster_smiles = [[combined_smiles[idx] for idx in cluster] for cluster in only_lib1]
            only_lib2_cluster_smiles = [[combined_smiles[idx] for idx in cluster] for cluster in only_lib2]
            return only_lib1_cluster_smiles, only_lib2_cluster_smiles
        else:
            raise ValueError("return_type must be 'all', 'mixed', 'only_lib1', 'only_lib2', or 'both_libs'.")
