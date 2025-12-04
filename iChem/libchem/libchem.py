import numpy as np # type: ignore

from ..bblean import pack_fingerprints, unpack_fingerprints
from ..utils import binary_fps
from ..bblean import BitBirch
from ..bblean.similarity import jt_isim_packed, estimate_jt_std, jt_isim_medoid, jt_stratified_sampling, jt_sim_matrix_between_packed
from ._libchem_aux import interiSIM, intraiSIM, MaxSum, MinSum

class LibChem:
    """Class for handling chemical libraries, including loading SMILES,
    generating fingerprints, calculating iSIM, and clustering molecules."""
    def __init__(self):
        self.smiles = None
        self.fps_packed = None
        self.n_molecules = 0

    def load_smiles(
            self, 
            smiles: str,
    ) -> None:
        """Load SMILES from a .smi file"""
        with open(smiles, 'r') as f:
            self.smiles = [line.split('\t', 1)[0].split(' ')[0].strip() for line in f if line.strip()]
        if self.fps_packed:
            if len(self.smiles) != self.n_molecules:
                raise ValueError("Number of SMILES does not match number of fingerprints.")
        self.n_molecules = len(self.smiles)

    def load_fingerprints(
            self,
            fingerprints : np.array, 
            packed: bool = True,
    ) -> None:
        """Load fingerprints from a numpy array"""
        self.n_molecules = fingerprints.shape[0]
        if packed:
            self.fps_packed = fingerprints
        else:
            self.fps_packed = pack_fingerprints(fingerprints)


    def generate_fingerprints(
            self,
            fp_type: str = 'ECFP4',
            n_bits: int = 2048,
    ) -> None:
        """Generate fingerprints from loaded SMILES"""
        if self.smiles is None:
            raise ValueError("SMILES data not loaded.")
        
        fps, _invalid = binary_fps(
            self.smiles,
            fp_type=fp_type,
            n_bits=n_bits,
            return_invalid=True,
        )

        self.fps_packed = pack_fingerprints(fps)
        self.n_molecules = len(fps)

        if len(_invalid) > 0:
            print(f"Warning: {len(_invalid)} invalid SMILES were skipped and deleted.")
            old_smiles = self.smiles
            self.smiles = np.delete(old_smiles, _invalid, axis=0).tolist()

    def get_fingerprints(
            self,
            packed: bool = True,
    ) -> np.array:
        """Retrieve fingerprints in packed or unpacked format"""
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")
        
        if packed:
            return self.fps_packed
        else:
            return unpack_fingerprints(self.fps_packed)
        
    def _calculate_iSIM(
            self,
    ) -> np.array:
        """Calculate iSIM similarity matrix from fingerprints"""
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")

        self.iSIM = jt_isim_packed(
            self.fps_packed,
        )

    def _calculate_iSIM_sigma(
            self,
            n_sigma_samples: int = 50
    ) -> np.array:
        """Calculate iSIM similarity matrix with sigma adjustment from fingerprints"""
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")
        
        self.iSIM_sigma = estimate_jt_std(
            self.fps_packed,
            n_samples=n_sigma_samples,
        )


    def get_iSIM(
            self,
            sim_index: str = 'JT',
    ) -> np.array:
        """Retrieve the calculated iSIM similarity matrix"""
        if not hasattr(self, 'iSIM'):
            self._calculate_iSIM()

        return self.iSIM
    
    def get_iSIM_sigma(
            self,
            n_sigma_samples: int = 50,
    ) -> float:
        """Retrieve the calculated iSIM sigma value"""
        if not hasattr(self, 'iSIM_sigma'):
            self._calculate_iSIM_sigma(n_sigma_samples=n_sigma_samples)
        return self.iSIM_sigma

    def _optimal_threshold(
            self,
            factor: float = 3.5,
    ) -> float:
        """Calculate the optimal threshold based on iSIM sigma"""
        if not hasattr(self, 'iSIM_sigma'):
            self._calculate_iSIM_sigma()
        if not hasattr(self, 'iSIM'):
            self._calculate_iSIM()

        return self.iSIM + factor * self.iSIM_sigma
    
    def set_threshold(
            self,
            threshold: float = None,
    ) -> None:
        """Set a custom optimal threshold for clustering"""
        if threshold is None:
            self.threshold = self._optimal_threshold()
        else:
            self.threshold = threshold
    
    def cluster(
            self,
            threshold: float = None,
            branching_factor: int = 1024,
            merge: str = 'diameter',
            recluster: bool = True,
    ) -> None:
        """Cluster molecules using BitBirch algorithm based on iSIM and optimal threshold"""
        if threshold:
            self.set_threshold(threshold)
        if self.threshold is None:
            self.set_threshold()
        
        bb_object = BitBirch(threshold=self.threshold,
                             branching_factor=branching_factor,
                             merge_criterion=merge)
        bb_object.fit(self.fps_packed)

        if not hasattr(self, 'iSIM_sigma'):
            self._calculate_iSIM_sigma()

        if recluster:
            bb_object.recluster_inplace(extra_threshold=self.iSIM_sigma, verbose=False, iterations=5)
        
        self.clusters = bb_object.get_cluster_mol_ids()

    def get_clusters(
            self,
    ) -> list[list[int]]:
        """Retrieve the clusters (indexes) formed after clustering"""
        if not hasattr(self, 'clusters'):
            self.cluster()
        
        return self.clusters
    
    def get_cluster_smiles(
            self,
    ) -> list[list[str]]:
        """Retrieve the SMILES strings for each cluster"""
        if self.smiles is None:
            raise ValueError("SMILES data not loaded.")
        if not hasattr(self, 'clusters'):
            self.cluster()
        
        cluster_smiles = []
        for cluster in self.clusters:
            cluster_smiles.append([self.smiles[i] for i in cluster])
        
        return cluster_smiles
    
    def get_cluster_medoids(
            self,
            return_smiles: bool = True,
    ) -> list[str]:
        """Retrieve the medoid SMILES string for each cluster"""
        if self.smiles is None:
            raise ValueError("SMILES data not loaded.")
        if not hasattr(self, 'clusters'):
            self.cluster()
        
        fingerprints_medoids = []
        if return_smiles:
            smiles_medoids = []
            for cluster in self.clusters:
                fps_cluster = self.fps_packed[cluster]
                medoid_index, medoid_fingerprint = jt_isim_medoid(fps_cluster)
                fingerprints_medoids.append(medoid_fingerprint)
                smiles_medoids.append(self.smiles[cluster[medoid_index]])
            
            return np.array(fingerprints_medoids), smiles_medoids
        else:
            for cluster in self.clusters:
                fps_cluster = self.fps_packed[cluster]
                medoid_index, medoid_fingerprint = jt_isim_medoid(fps_cluster)
                fingerprints_medoids.append(medoid_fingerprint)
            
            return np.array(fingerprints_medoids)

    def get_cluster_samples(
            self,
            n_samples: int = 1000,
            return_smiles: bool = False,
            return_cluster_ids: bool = False,
        ) -> None:
        """Cluster a sample of molecules using stratified sampling"""
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")
        if self.smiles is None and return_smiles is True:
            raise ValueError("SMILES data not loaded.")
        if self.n_molecules < n_samples:
            raise ValueError("Number of samples exceeds number of molecules.")
        if not hasattr(self, 'clusters'):
            self.cluster()
        if len(self.clusters) == n_samples:
            if return_cluster_ids and return_smiles:
                medoids, smiles = self.get_cluster_medoids(return_smiles=True)
                cluster_ids = list(range(len(self.clusters)))
                return medoids, smiles, cluster_ids
            if return_cluster_ids:
                cluster_ids = list(range(len(self.clusters)))
                medoids = self.get_cluster_medoids(return_smiles=False)
                return medoids, cluster_ids
            if return_smiles:
                medoids, smiles = self.get_cluster_medoids(return_smiles=True)
                return medoids, smiles
            return self.get_cluster_medoids(return_smiles=return_smiles)
        
        sampled_fps = []
        sampled_smiles = []
        cluster_ids = []
        # Compute the medoids (which will be included in the sample for sure)
        medoids_fps, medoids_smiles = self.get_cluster_medoids(return_smiles=True)
        # First case: number of clusters is larger than n_samples
        if len(self.clusters) >= n_samples:
            sampled_fps = medoids_fps[:n_samples]
            if return_smiles:
                sampled_smiles = medoids_smiles[:n_samples]
            if return_cluster_ids:
                cluster_ids = list(range(n_samples))
        else:
            # Second case: number of clusters is smaller than n_samples
            # include all medoids first
            sampled_fps.extend(medoids_fps)
            if return_smiles:
                sampled_smiles.extend(medoids_smiles)
            if return_cluster_ids:
                cluster_ids.extend(list(range(len(self.clusters))))

            # Update the number of samples to draw
            n_samples_updated = n_samples - len(self.clusters)

            # Stratified sampling from each cluster
            for k, cluster in enumerate(self.clusters):
                n_cluster = len(cluster)
                fps_cluster = self.fps_packed[cluster]
                n_sample_cluster = max(1, int(n_cluster * n_samples / self.n_molecules))
                if n_samples_updated < n_sample_cluster - 1:
                    n_sample_cluster = n_samples_updated + 1
                sampled_indices = jt_stratified_sampling(fps_cluster, n_samples=n_sample_cluster)
                n_samples_updated -= (n_sample_cluster - 1)
                sampled_indices = sampled_indices[1:]

                sampled_fps_ = fps_cluster[sampled_indices]
                sampled_fps.extend(sampled_fps_)
                
                if return_smiles:
                    sampled_smiles_ = [self.smiles[cluster[i]] for i in sampled_indices]
                    sampled_smiles.extend(sampled_smiles_)
                if return_cluster_ids:
                    cluster_ids.extend([k] * len(sampled_indices))
                    
                if n_samples_updated <= 0:
                    break

        if return_cluster_ids and return_smiles:
            return np.array(sampled_fps), sampled_smiles, cluster_ids
        elif return_smiles:
            return np.array(sampled_fps), sampled_smiles
        elif return_cluster_ids:
            return np.array(sampled_fps), cluster_ids
        else:
            return np.array(sampled_fps)

class LibComparison:
    def __init__(self, 
    ):
        self.libraries = {}

    def add_library(self,
            library: LibChem,
            lib_name: str,
    ) -> None:
        """Add a library to the comparison"""
        self.libraries[lib_name] = library

    def compare_libraries(self,
                        methodology: str = 'intraiSIM',
                        lib1_name: str = None,
                        lib2_name: str = None) -> float:
        """Compare two libraries using specified methodology"""
        if lib1_name == None and lib2_name == None and len(self.libraries) == 2:
            lib1_name, lib2_name = list(self.libraries.keys())
        if lib1_name not in list(self.libraries.keys()) or lib2_name not in list(self.libraries.keys()):
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
        if lib1_name not in list(self.libraries.keys()) or lib2_name not in list(self.libraries.keys()):
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


    def _cluster_medoid_mix(self,
                            threshold: float = None,
                            lib_names: list[str] = None,
                            verbose: bool = False) -> float:
        """Cluster the combined medoids of both libraries"""
        if lib_names is None:
            lib_names = list(self.libraries.keys())
        if len(lib_names) < 2:
            raise ValueError("At least two libraries must be specified for comparison.")
        
        medoids_fps = []
        medoids_smiles = []
        medoids_flags = []
        for lib_name in lib_names:
            if lib_name not in list(self.libraries.keys()):
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

        # Do the clustering
        combined_lib.cluster(threshold=threshold, indices=indices.tolist())

        # Set the fingerprints and smiles to the combined library
        combined_lib.fps_packed = pack_fingerprints(medoids_fps)
    
        # Get the clusters
        clusters = combined_lib.get_clusters()
    
        return clusters, medoids_smiles, medoids_flags
    
    def _cluster_sample_mix(self,
                           n_samples: int = None,
                           threshold: float = None,
                           lib_names: list[str] = None,
                           verbose: bool = False) -> float:
        """Cluster a sample of combined samples of both libraries"""
        if self.lib1_name is None and self.lib2_name is None and len(self.libraries) == 2:
            pass
        fps1 = self.lib1.cluster_sample(n_samples=n_samples, return_smiles=False)[0]
        fps2 = self.lib2.cluster_sample(n_samples=n_samples, return_smiles=False)[0]

        n1 = len(fps1)
        n2 = len(fps2)
        n3 = n1 + n2
        print(f'Number of sampled mols in Library 1: {n1}, proportion: {n1/n3:.2f}')
        print(f'Number of sampled mols in Library 2: {n2}, proportion: {n2/n3:.2f}')
        print(f'Total number of sampled mols: {n3}')

        # Create a new LibChem instance for combined sampled mols
        combined_lib = LibChem()
        fps3 = np.array(fps1 + fps2)
        indices = np.arange(n3)
        np.random.shuffle(indices)
        fps3 = fps3[indices]
        if self.lib1.smiles is not None and self.lib2.smiles is not None:
            combined_smiles = self.lib1.smiles + self.lib2.smiles
            combined_lib.smiles = [combined_smiles[i] for i in indices]
        combined_lib.fps_packed = pack_fingerprints(fps3)
        combined_lib.n_molecules = len(fps3)

        # Delete temporary variables to free memory
        del fps1, fps2

        # Perform clustering on the combined library
        combined_lib.cluster(indices=indices.tolist())

        # Get the clusters
        clusters = combined_lib.get_clusters()

        return clusters, n1
    
    def cluster_medoids_proportion(self,
                           threshold: float = None,) -> float:
        """Cluster the combined medoids of both libraries"""
        clusters, n1 = self._cluster_medoid_mix(threshold=threshold)

        mixed_cluster_count, only_lib1_count, only_lib2_count = cluster_class_counts(clusters, n1=n1)

        proportion_mixed, proportion_1, proportion_2 = calc_proportions(mixed_cluster_count, only_lib1_count, only_lib2_count)
        return proportion_mixed, proportion_1, proportion_2
    
    def cluster_sample_proportions(self,
                           n_samples: int = None,) -> float:
        """Cluster a sample of combined medoids of both libraries"""
        if n_samples is None:
            raise ValueError("Number of samples must be specified.")

        # Get the clusters
        clusters, n1 = self._cluster_sample_mix(n_samples=n_samples)

        # Calculate the proportion of mixed clusters
        mixed_cluster_count, only_lib1_count, only_lib2_count = cluster_class_counts(clusters, n1=n1)
        proportion_mixed, proportion_1, proportion_2 = calc_proportions(mixed_cluster_count, only_lib1_count, only_lib2_count)
        return proportion_mixed, proportion_1, proportion_2
    
    def get_combined_lib_clusters(self,
                                  methodology: str = 'samples',
                                  n_samples: int = None,
                                  return_type: str = 'all',) -> list[list[int]]:
        
        """Get clusters from combined libraries based on specified methodology"""
        if methodology == 'medoids':
            clusters, n1 = self._cluster_medoid_mix()
        elif methodology == 'samples':
            if n_samples is None:
                raise ValueError("Number of samples must be specified for 'samples' methodology.")
            clusters, n1 = self._cluster_sample_mix(n_samples=n_samples)
        else:
            raise ValueError("Methodology must be either 'medoids' or 'samples'.")
        
        combined_smiles = self.lib1.smiles + self.lib2.smiles
        mixed, only_lib1, only_lib2 = cluster_classification(clusters, n1=n1)

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

    def _cluster_all_together(self
                              ) -> None:
        """Cluster all molecules from both libraries together"""
        fps1 = self.lib1.get_fingerprints(packed=True)
        n1 = len(fps1)
        fps2 = self.lib2.get_fingerprints(packed=True)
        fps3 = np.vstack((fps1, fps2))
        indices = np.arange(len(fps3))
        np.random.shuffle(indices)
        fps3 = fps3[indices]

        combined_lib = LibChem()
        combined_lib.load_fingerprints(
            fingerprints = fps3,
            packed = True,
        )

        combined_lib.cluster()
        clusters = combined_lib.get_clusters()

        return clusters, n1
    
    def cluster_all_proportions(self
                              ) -> float:
        """Cluster all molecules from both libraries together and calculate proportions"""
        clusters, n1 = self._cluster_all_together()

        mixed_cluster_count, only_lib1_count, only_lib2_count = cluster_class_counts(clusters, n1=n1)

        proportion_mixed, proportion_1, proportion_2 = calc_proportions(mixed_cluster_count, only_lib1_count, only_lib2_count)
        return proportion_mixed, proportion_1, proportion_2
    
def cluster_class_counts(clusters: list[list[int]], n1: int) -> dict:
    """Count the number of clusters by class composition"""
    mixed = 0
    only_lib1 = 0
    only_lib2 = 0
    for cluster in clusters:
        has_lib1 = all(idx < n1 for idx in cluster)
        has_lib2 = all(idx >= n1 for idx in cluster)
        if has_lib1:
            only_lib1 += 1
        elif has_lib2:
            only_lib2 += 1
        else:
            mixed += 1
    return mixed, only_lib1, only_lib2

def cluster_classification(clusters: list[list[int]], n1: int) -> list[str]:
    """Classify clusters based on their composition"""
    mixed, only_lib1, only_lib2 = [], [], []
    for k, cluster in enumerate(clusters):
        has_lib1 = all(idx < n1 for idx in cluster)
        has_lib2 = all(idx >= n1 for idx in cluster)
        if has_lib1:
            only_lib1.append(k)
        elif has_lib2:
            only_lib2.append(k)
        else:
            mixed.append(k)
    return [clusters[i] for i in mixed], [clusters[i] for i in only_lib1], [clusters[i] for i in only_lib2]

def calc_proportions(mixed, only_1, only_2):
    """Calculate proportions of cluster types"""
    total = mixed + only_1 + only_2
    return mixed / total, only_1 / total, only_2 / total