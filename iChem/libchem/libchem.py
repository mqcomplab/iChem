import numpy as np # type: ignore

from ..bblean import pack_fingerprints, unpack_fingerprints
from ..utils import binary_fps
from ..bblean import BitBirch
from ..bblean.similarity import jt_isim_packed, estimate_jt_std, jt_isim_medoid, jt_stratified_sampling

class LibChem:
    """Class for handling chemical libraries, including loading SMILES,
    generating fingerprints, calculating iSIM, and clustering molecules."""
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
        """Load SMILES from a .smi file
        
        Args:
            smiles (str): Path to the .smi file containing SMILES strings.
        """
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
        """Load fingerprints from a numpy array
        
        Args:
            fingerprints (np.array): Numpy array of fingerprints.
            packed (bool): Whether the fingerprints are in packed format. Defaults to True.
        """
        self.n_molecules = fingerprints.shape[0]
        if packed:
            self.fps_packed = fingerprints
        else:
            self.fps_packed = pack_fingerprints(fingerprints)

    def load_flags(self,
                   flags: list,
     ) -> None:
          """Load flags associated with each molecule"""
          if len(flags) != self.n_molecules:
                raise ValueError("Number of flags does not match number of molecules.")
          self.flags = flags

    def generate_fingerprints(
            self,
            fp_type: str = 'ECFP4',
            n_bits: int = 2048,
    ) -> None:
        """Generate fingerprints from loaded SMILES. Saves the fingerprints in the class instance.
        
        Args:
            fp_type (str): Type of fingerprint to generate. Defaults to 'ECFP4'. Other types: 'ECFP6', 'AP', 'RDKIT'.
            n_bits (int): Number of bits for the fingerprint. Defaults to 2048.
        """
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
        """Retrieve fingerprints in packed or unpacked format
        
        Args:
            packed (bool): Whether to return fingerprints in packed format. Defaults to True.

        Returns:
            np.array: Numpy array of fingerprints.
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
        """Retrieve the flags associated with each molecule
        
        Returns:
            list: List of flags.
        """
        if self.flags is None:
            raise ValueError("Flags not loaded.")
        return self.flags

    def _calculate_iSIM(
            self,
    ) -> float:
        """Calculate iSIM similarity matrix from fingerprints
        
        Returns:
        float: iSIM similarity matrix.
        """
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")

        self.iSIM = jt_isim_packed(
            self.fps_packed,
        )

    def _calculate_iSIM_sigma(
            self,
            n_sigma_samples: int = 50
    ) -> float:
        """Calculate iSIM similarity matrix with sigma adjustment from fingerprints. Values are stored in the class instance.
        
        Args:
            n_sigma_samples (int): Number of samples to estimate sigma. Defaults to 50.
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
        """Retrieve the calculated iSIM similarity matrix

        Returns:
            float: iSIM Tanimotosimilarity value. 
        """
        if not hasattr(self, 'iSIM'):
            self._calculate_iSIM()

        return self.iSIM
    
    def get_iSIM_sigma(
            self,
            n_sigma_samples: int = 50,
    ) -> float:
        """Retrieve the calculated iSIM sigma value

        Args:
            n_sigma_samples (int): Number of samples to estimate sigma. Defaults to 50.

        Returns:
            float: iSIM sigma value.
        """
        if not hasattr(self, 'iSIM_sigma'):
            self._calculate_iSIM_sigma(n_sigma_samples=n_sigma_samples)
        return self.iSIM_sigma

    def _optimal_threshold(
            self,
            factor: float = 3.5,
    ) -> float:
        """Calculate the optimal threshold based on iSIM sigma
        
        Args:
            factor (float): Multiplicative factor for sigma. Defaults to 3.5.
            
        Returns:
            float: Optimal threshold value.
        """
        if not hasattr(self, 'iSIM_sigma'):
            self._calculate_iSIM_sigma()
        if not hasattr(self, 'iSIM'):
            self._calculate_iSIM()

        return self.iSIM + factor * self.iSIM_sigma
    
    def set_threshold(
            self,
            threshold: float = None,
            factor : float = 3.5,
    ) -> None:
        """Set a custom optimal threshold for clustering
        Args:
            threshold (float): Custom threshold value. If None, calculates optimal threshold.
            factor (float): Multiplicative factor for sigma when calculating optimal threshold. Defaults to 3.5.
        """
        if threshold is None:
            self.threshold = self._optimal_threshold(factor=factor)
        else:
            self.threshold = threshold
    
    def cluster(
            self,
            threshold: float = None,
            threshold_factor: float = 3.5,
            branching_factor: int = 1024,
            merge: str = 'diameter',
            recluster: bool = True,
    ) -> None:
        """Cluster molecules using BitBirch algorithm based on iSIM and optimal threshold.
        Clustered indexes are stored in the class instance.
        
        Args:
            threshold (float): Custom threshold value for clustering. If None, uses optimal threshold.
            threshold_factor (float): Multiplicative factor for sigma when calculating optimal threshold. Defaults to 3.5.
            branching_factor (int): Branching factor for BitBirch algorithm. Defaults to 1024.
            merge (str): Merge criterion for BitBirch algorithm. Defaults to 'diameter'.
            recluster (bool): Whether to perform reclustering step. Defaults to True.
        """
        if threshold:
            self.set_threshold(threshold)
        if self.threshold is None:
            self.set_threshold(factor=threshold_factor)
        
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
        """Retrieve the clusters (indexes) formed after clustering
        Returns:
            list[list[int]]: List of clusters with molecule indexes.
        """
        if not hasattr(self, 'clusters'):
            self.cluster()
        
        return self.clusters
    
    def get_cluster_flags(self) -> list[list[str]]:
        """Retrieve the flags for each cluster
        Returns:
            list[list[str]]: List of clusters with flags.
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
        """Retrieve the SMILES strings for each cluster
        
        Returns:
            list[list[str]]: List of clusters with SMILES strings.
        """
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
        """Retrieve the medoid SMILES string for each cluster
        
        Args:
            return_smiles (bool): Whether to return SMILES strings along with fingerprints. Defaults to True.
            
        Returns:
            list[str]: List of medoid fingerprints or (fingerprints, SMILES) tuples.
        """
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
        """Cluster a sample of molecules using stratified sampling from each cluster.
        
        Args:
            n_samples (int): Number of samples to draw. Defaults to 1000.
            return_smiles (bool): Whether to return SMILES strings along with fingerprints. Defaults to False.
            return_cluster_ids (bool): Whether to return cluster IDs along with fingerprints. Defaults to False.
        Returns:
            np.array: Sampled fingerprints.
            list[str] (optional): Sampled SMILES strings.
            list[int] (optional): Cluster IDs for each sampled molecule.
        """
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