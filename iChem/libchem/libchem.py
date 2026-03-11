import numpy as np # type: ignore

from ..bblean import pack_fingerprints, unpack_fingerprints
from ..utils import binary_fps
from ..utils import load_smiles as _load_smiles
from ..bblean import BitBirch
from ..bblean.similarity import jt_isim_packed, estimate_jt_std, jt_isim_medoid, jt_stratified_sampling

class LibChem:
    """Class for handling chemical libraries, including loading SMILES,
    generating fingerprints, calculating iSIM, and clustering molecules.
    
    This class provides a comprehensive interface for working with chemical libraries,
    including molecular fingerprint generation, similarity calculations using iSIM
    (intrinsic SIMilarity), and hierarchical clustering using the BitBirch algorithm.
    
    Attributes:
        smiles (list): List of SMILES strings representing molecules.
        fps_packed (np.ndarray): Packed binary fingerprints of molecules.
        n_molecules (int): Number of molecules in the library.
        threshold (float): Clustering threshold value.
        flags (list): Optional flags associated with each molecule.
        iSIM (float): Calculated intrinsic similarity value.
        iSIM_sigma (float): Standard deviation of iSIM.
        clusters (list): List of clusters containing molecule indices.
        cluster_medoids_fps (np.ndarray): Fingerprints of cluster medoids.
        cluster_medoids_smiles (list): SMILES strings of cluster medoids.
        sample_cluster_fps (np.ndarray): Sampled fingerprints from clusters.
        sample_cluster_smiles (list): SMILES strings of sampled molecules.
        sample_cluster_ids (list): Cluster IDs for sampled molecules.
    """
    def __init__(self):
        self.smiles = None
        self.fps_packed = None
        self.n_molecules = 0
        self.threshold = None
        self.flags = None

    def load_smiles(
            self, 
            smiles: str,
    ) -> None:
        """Load SMILES strings from a .smi file.
        
        Reads SMILES strings from a file and stores them in the class instance.
        Updates the molecule count and validates consistency with fingerprints if present.
        
        Args:
            smiles (str): Path to the .smi file containing SMILES strings.
            
        Raises:
            ValueError: If the number of SMILES doesn't match existing fingerprints.
        """
        self.smiles = _load_smiles(smiles)
        if self.fps_packed is not None:
            if len(self.smiles) != self.n_molecules:
                raise ValueError("Number of SMILES does not match number of fingerprints.")
        self.n_molecules = len(self.smiles)

    def set_smiles(
            self,
            smiles: list,
    ) -> None:
        """Set SMILES strings directly from a list.
        
        Assigns SMILES strings from a list and updates the molecule count.
        Validates consistency with fingerprints if present.
        
        Args:
            smiles (list): List of SMILES strings representing molecules.
            
        Raises:
            ValueError: If the number of SMILES doesn't match existing fingerprints.
        """
        self.smiles = smiles
        if self.fps_packed is not None:
            if len(self.smiles) != self.n_molecules:
                raise ValueError("Number of SMILES does not match number of fingerprints.")
        self.n_molecules = len(self.smiles)

    def set_fingerprints(
            self,
            fingerprints : np.array, 
            packed: bool = True,
    ) -> None:
        """Set molecular fingerprints from a numpy array.
        
        Assigns fingerprints to the class instance and updates molecule count.
        If fingerprints are not packed, they will be packed automatically.
        
        Args:
            fingerprints (np.ndarray): Numpy array of molecular fingerprints.
            packed (bool, optional): Whether the fingerprints are in packed format. 
                Defaults to True. If False, fingerprints will be packed.
        """
        self.n_molecules = fingerprints.shape[0]
        if packed:
            self.fps_packed = fingerprints
        else:
            self.fps_packed = pack_fingerprints(fingerprints)

    def load_fingerprints(
            self,
            fingerprints: np.array,
            packed: bool = True,
    ) -> None:
        """Load molecular fingerprints from a .npy file.
        
        Reads fingerprints from a numpy file using memory mapping and stores them
        in the class instance. The file is closed after loading to free memory.

        Args:
            fingerprints (str): Path to the .npy file containing fingerprints.
            packed (bool, optional): Whether the fingerprints are in packed format. 
                Defaults to True. If False, fingerprints will be packed.
        """
        fingerprints = np.load(fingerprints, mmap_mode='r')
        self.set_fingerprints(fingerprints, packed=packed)
        del fingerprints

    def set_flags(self,
                   flags: list,
     ) -> None:
          """Set flags associated with each molecule.
          
          Flags can be used to store metadata or categorical information about molecules.
          
          Args:
              flags (list): List of flags, one per molecule.
              
          Raises:
              ValueError: If the number of flags doesn't match the number of molecules.
          """
          if len(flags) != self.n_molecules:
                raise ValueError("Number of flags does not match number of molecules.")
          self.flags = flags

    def generate_fingerprints(
            self,
            fp_type: str = 'ECFP4',
            n_bits: int = 2048,
    ) -> None:
        """Generate molecular fingerprints from loaded SMILES strings.
        
        Generates binary fingerprints in packed format and stores them in the class instance.
        Invalid SMILES are automatically skipped and removed from the SMILES list.
        
        Args:
            fp_type (str, optional): Type of fingerprint to generate. Defaults to 'ECFP4'.
                Supported types: 'ECFP4', 'ECFP6', 'AP', 'RDKIT', 'MACCS'.
            n_bits (int, optional): Number of bits for the fingerprint. Defaults to 2048.
                Only applicable for certain fingerprint types.
                
        Raises:
            ValueError: If SMILES data has not been loaded.
            
        Note:
            Invalid SMILES will be skipped and a warning message will be printed.
        """
        if self.smiles is None:
            raise ValueError("SMILES data not loaded.")
        
        fps, _invalid = binary_fps(
            self.smiles,
            fp_type=fp_type,
            n_bits=n_bits,
            return_invalid=True,
            packed=True
        )

        self.fps_packed = fps
        self.n_molecules = len(fps)

        if len(_invalid) > 0:
            print(f"Warning: {len(_invalid)} invalid SMILES were skipped and deleted.")
            old_smiles = self.smiles
            self.smiles = np.delete(old_smiles, _invalid, axis=0).tolist()

    def get_fingerprints(
            self,
            packed: bool = True,
    ) -> np.array:
        """Retrieve molecular fingerprints in packed or unpacked format.
        
        Args:
            packed (bool, optional): Whether to return fingerprints in packed format. 
                Defaults to True. If False, returns unpacked binary fingerprints.

        Returns:
            np.ndarray: Numpy array of molecular fingerprints in the requested format.
            
        Raises:
            ValueError: If fingerprints have not been loaded or generated.
        """
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")
        
        if packed:
            return self.fps_packed
        else:
            return unpack_fingerprints(self.fps_packed)
        
    def get_flags(
            self,
    ) -> list:
        """Retrieve the flags associated with each molecule.
        
        Returns:
            list: List of flags, one per molecule.
            
        Raises:
            ValueError: If flags have not been loaded.
        """
        if self.flags is None:
            raise ValueError("Flags not loaded.")
        return self.flags

    def _calculate_iSIM(
            self,
    ) -> None:
        """Calculate intrinsic similarity (iSIM) from fingerprints.
        
        Computes the intrinsic Tanimoto similarity value for the molecular library
        and stores it in the class instance.
        
        Raises:
            ValueError: If fingerprints have not been loaded or generated.
            
        Note:
            This is an internal method. Use get_iSIM() to retrieve the value.
        """
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")

        self.iSIM = jt_isim_packed(
            self.fps_packed,
        )

    def _calculate_iSIM_sigma(
            self,
            n_sigma_samples: int = 50
    ) -> None:
        """Calculate the standard deviation (sigma) of iSIM.
        
        Estimates the standard deviation of the intrinsic Tanimoto similarity
        distribution and stores it in the class instance.
        
        Args:
            n_sigma_samples (int, optional): Number of samples to use for estimating sigma. 
                Defaults to 50.
                
        Raises:
            ValueError: If fingerprints have not been loaded or generated.
            
        Note:
            This is an internal method. Use get_iSIM_sigma() to retrieve the value.
        """
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")
        
        self.iSIM_sigma = estimate_jt_std(
            self.fps_packed,
            n_samples=n_sigma_samples,
        )

    def get_iSIM(
            self,
    ) -> float:
        """Retrieve the intrinsic similarity (iSIM) value.
        
        If iSIM has not been calculated, it will be computed automatically.

        Returns:
            float: Intrinsic Tanimoto similarity value for the molecular library.
        """
        if not hasattr(self, 'iSIM'):
            self._calculate_iSIM()

        return self.iSIM
    
    def get_iSIM_sigma(
            self,
            n_sigma_samples: int = 50,
    ) -> float:
        """Retrieve the standard deviation (sigma) of iSIM.
        
        If sigma has not been calculated, it will be computed automatically.

        Args:
            n_sigma_samples (int, optional): Number of samples to estimate sigma. 
                Defaults to 50. Only used if sigma hasn't been calculated yet.

        Returns:
            float: Standard deviation of intrinsic Tanimoto similarity.
        """
        if not hasattr(self, 'iSIM_sigma'):
            self._calculate_iSIM_sigma(n_sigma_samples=n_sigma_samples)
        return self.iSIM_sigma

    def _optimal_threshold(
            self,
            factor: float = 3.5,
    ) -> float:
        """Calculate the optimal clustering threshold based on iSIM and sigma.
        
        The optimal threshold is computed as: iSIM + factor * sigma
        
        Args:
            factor (float, optional): Multiplicative factor for sigma. Defaults to 3.5.
                Higher values result in stricter clustering (fewer, smaller clusters).
            
        Returns:
            float: Optimal threshold value for clustering.
            
        Note:
            This is an internal method. Use set_threshold() to set the threshold.
        """
        if not hasattr(self, 'iSIM_sigma'):
            self._calculate_iSIM_sigma()
        if not hasattr(self, 'iSIM'):
            self._calculate_iSIM()

        return self.iSIM + factor * self.iSIM_sigma
    
    def set_threshold(
            self,
            threshold: float = None,
            factor : float = None,
    ) -> None:
        """Set the clustering threshold.
        
        Sets either a custom threshold value or calculates the optimal threshold
        based on iSIM statistics.
        
        Args:
            threshold (float, optional): Custom threshold value. If None, calculates 
                optimal threshold based on iSIM and sigma. Defaults to None.
            factor (float, optional): Multiplicative factor for sigma when calculating 
                optimal threshold. Defaults to 3.5. Higher values result in stricter clustering.
        """
        if threshold is None and self.threshold is None:
            self.threshold = self._optimal_threshold(factor=3.5 if factor is None else factor)
        else:
            self.threshold = threshold
    
    def cluster(
            self,
            threshold: float = None,
            factor: float = None,
            branching_factor: int = 1024,
            merge: str = 'diameter',
            recluster: bool = True,
    ) -> None:
        """Cluster molecules using the BitBirch hierarchical clustering algorithm.
        
        Performs clustering of molecules based on fingerprint similarity
        using the BitBirch algorithm. Cluster assignments are stored in the class instance.
        
        Args:
            threshold (float, optional): Custom threshold value for clustering. If None, 
                uses the optimal threshold based on iSIM statistics. Defaults to None.
            factor (float, optional): Multiplicative factor for sigma when calculating 
                optimal threshold. Defaults to 3.5. Only used if threshold is None.
            branching_factor (int, optional): Branching factor for BitBirch algorithm. 
                Controls tree structure. Defaults to 1024.
            merge (str, optional): Merge criterion for BitBirch algorithm. 
                Options: 'diameter', 'centroid'. Defaults to 'diameter'.
            recluster (bool, optional): Whether to perform iterative reclustering step 
                to refine clusters. Defaults to True.
        """
        if self.threshold is None and threshold is None:
            self.set_threshold()
        elif threshold is not None and self.threshold is None:
            self.set_threshold(threshold=threshold)
        elif threshold is None and factor is not None:
            self.set_threshold(factor=factor)
        
        bb_object = BitBirch(threshold=self.threshold,
                             branching_factor=branching_factor,
                             merge_criterion=merge)
        bb_object.fit(self.fps_packed)

        if recluster:
            if not hasattr(self, 'iSIM_sigma'):
                self._calculate_iSIM_sigma()
            bb_object.recluster_inplace(extra_threshold=self.iSIM_sigma, verbose=False, iterations=5)
        
        self.clusters = bb_object.get_cluster_mol_ids()

    def get_clusters(
            self,
    ) -> list[list[int]]:
        """Retrieve the cluster assignments.
        
        Returns molecule indices grouped by cluster. If clustering has not been
        performed, it will be executed automatically.
        
        Returns:
            list[list[int]]: List of clusters, where each cluster is a list of 
                molecule indices belonging to that cluster.
        """
        if not hasattr(self, 'clusters'):
            self.cluster()
        
        return self.clusters
    
    def get_cluster_flags(self) -> list[list[str]]:
        """Retrieve the flags for each cluster.
        
        Returns flags grouped by cluster assignment. If clustering has not been
        performed, it will be executed automatically.
        
        Returns:
            list[list[str]]: List of clusters, where each cluster is a list of 
                flags corresponding to molecules in that cluster.
                
        Raises:
            ValueError: If flags have not been loaded.
        """
        if not self.flags:
            raise ValueError("Flags not loaded.")
        if not hasattr(self, 'clusters'):
            self.cluster()
        
        flags = self.flags
        cluster_flags = []
        for cluster in self.clusters:
            cluster_flags.append([flags[i] for i in cluster])
        
        return cluster_flags
    
    def get_cluster_smiles(
            self,
    ) -> list[list[str]]:
        """Retrieve the SMILES strings for each cluster.
        
        Returns SMILES strings grouped by cluster assignment. If clustering has not
        been performed, it will be executed automatically.
        
        Returns:
            list[list[str]]: List of clusters, where each cluster is a list of 
                SMILES strings corresponding to molecules in that cluster.
                
        Raises:
            ValueError: If SMILES data has not been loaded.
        """
        if self.smiles is None:
            raise ValueError("SMILES data not loaded.")
        if not hasattr(self, 'clusters'):
            self.cluster()
        
        cluster_smiles = []
        for cluster in self.clusters:
            cluster_smiles.append([self.smiles[i] for i in cluster])
        
        return cluster_smiles
    
    def save_cluster_medoids(
            self,
    ) -> None:
        """Calculate and save the medoid for each cluster.
        
        Computes the medoid (most representative molecule) for each cluster based
        on fingerprint similarity. Stores both fingerprints and SMILES strings of
        medoids in the class instance.
        
        Raises:
            ValueError: If SMILES data has not been loaded or clustering has not been performed.
            
        Note:
            Medoids are stored in cluster_medoids_fps and cluster_medoids_smiles attributes.
        """
        if self.smiles is None:
            raise ValueError("SMILES data not loaded.")
        if not hasattr(self, 'clusters'):
            raise ValueError("Clustering not performed. Please run cluster() method first. Or set_threshold() and run cluster() method.")

        fingerprints_medoids = []
        smiles_medoids = []
        for cluster in self.clusters:
            fps_cluster = self.fps_packed[cluster]
            medoid_index, medoid_fingerprint = jt_isim_medoid(fps_cluster)
            fingerprints_medoids.append(medoid_fingerprint)
            smiles_medoids.append(self.smiles[cluster[medoid_index]])
        
        self.cluster_medoids_fps = np.array(fingerprints_medoids)
        self.cluster_medoids_smiles = smiles_medoids

        
    def get_cluster_medoids(
            self,
            return_smiles: bool = True,
    ) -> tuple:
        """Retrieve the medoid fingerprints and/or SMILES for each cluster.
        
        Returns the most representative molecule (medoid) for each cluster.
        If medoids have not been calculated, they will be computed automatically.
        
        Args:
            return_smiles (bool, optional): Whether to return SMILES strings along with 
                fingerprints. Defaults to True.
            
        Returns:
            np.ndarray or tuple: If return_smiles is True, returns a tuple of 
                (fingerprints, smiles). If False, returns only fingerprints array.
        """

        if not hasattr(self, 'cluster_medoids_fps'):
            self.save_cluster_medoids()

        return (self.cluster_medoids_fps, self.cluster_medoids_smiles) if return_smiles else self.cluster_medoids_fps
        
    def save_cluster_samples(
            self,
            n_samples: int = 1000
        ) -> None:
        """Sample molecules from clusters using stratified sampling.
        
        Performs stratified sampling to select representative molecules from each cluster.
        Medoids are always included in the sample. Samples are stored in the class instance.
        
        Args:
            n_samples (int, optional): Total number of samples to draw. Defaults to 1000.
                Must not exceed the total number of molecules.
                
        Raises:
            ValueError: If fingerprints, SMILES are not loaded, or if n_samples exceeds
                the number of molecules, or if clustering has not been performed.
                
        Note:
            Samples are stored in sample_cluster_fps, sample_cluster_smiles, and
            sample_cluster_ids attributes.
        """
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")
        if self.smiles is None:
            raise ValueError("SMILES data not loaded.")
        if self.n_molecules < n_samples:
            raise ValueError("Number of samples exceeds number of molecules.")
        if not hasattr(self, 'clusters'):
            raise ValueError("Clustering not performed. Please run cluster() method first. Or set_threshold() and run cluster() method.")
        
        if len(self.clusters) == n_samples:
                medoids, smiles = self.get_cluster_medoids(return_smiles=True)
                cluster_ids = list(range(len(self.clusters)))
                return medoids, smiles, cluster_ids
        
        sampled_fps = []
        sampled_smiles = []
        cluster_ids = []
        # Compute the medoids (which will be included in the sample for sure)
        medoids_fps, medoids_smiles = self.get_cluster_medoids(return_smiles=True)
        # First case: number of clusters is larger than n_samples
        if len(self.clusters) >= n_samples:
            sampled_fps = medoids_fps[:n_samples]
            sampled_smiles = medoids_smiles[:n_samples]
            cluster_ids = list(range(n_samples))
        else:
            # Second case: number of clusters is smaller than n_samples
            # include all medoids first
            sampled_fps.extend(medoids_fps)
            sampled_smiles.extend(medoids_smiles)
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
                
                sampled_smiles_ = [self.smiles[cluster[i]] for i in sampled_indices]
                sampled_smiles.extend(sampled_smiles_)

                cluster_ids.extend([k] * len(sampled_indices))
                    
                if n_samples_updated <= 0:
                    break


        self.sample_cluster_fps = np.array(sampled_fps)
        self.sample_cluster_smiles = sampled_smiles
        self.sample_cluster_ids = cluster_ids

        
    def get_cluster_samples(
            self,
            n_samples: int = 1000,
            return_smiles: bool = False,
            return_cluster_ids: bool = False,
        ) -> np.array:
        """Retrieve sampled molecules from clusters using stratified sampling.
        
        Returns representative samples from each cluster. If samples have not been
        generated, they will be computed automatically.
        
        Args:
            n_samples (int, optional): Total number of samples to draw. Defaults to 1000.
            return_smiles (bool, optional): Whether to return SMILES strings along with 
                fingerprints. Defaults to False.
            return_cluster_ids (bool, optional): Whether to return cluster IDs for each
                sampled molecule. Defaults to False.
                
        Returns:
            np.ndarray or tuple: Returns data based on the flags:
                - If both flags are False: returns only fingerprints (np.ndarray)
                - If return_smiles is True: returns (fingerprints, smiles)
                - If return_cluster_ids is True: returns (fingerprints, cluster_ids)
                - If both are True: returns (fingerprints, smiles, cluster_ids)
        """

        if not hasattr(self, 'sample_cluster_fps'):
            self.save_cluster_samples(n_samples=n_samples)

        if return_cluster_ids and return_smiles:
            return self.sample_cluster_fps, self.sample_cluster_smiles, self.sample_cluster_ids
        if return_cluster_ids:
            return self.sample_cluster_fps, self.sample_cluster_ids
        if return_smiles:
            return self.sample_cluster_fps, self.sample_cluster_smiles
        return self.sample_cluster_fps

    
    def empty_fps(
            self,
    ) -> None:
        """Remove fingerprints from memory to free up resources.
        
        Sets the fingerprints attribute to None, allowing the memory to be freed
        by garbage collection. Useful for large libraries when fingerprints are
        no longer needed.
        """
        self.fps_packed = None