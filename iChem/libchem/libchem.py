import numpy as np # type: ignore

from ..bblean import pack_fingerprints, unpack_fingerprints
from ..utils import binary_fps
from ..bblean import BitBirch
from ..iSIM import calculate_isim, calculate_medoid
from ..iSIM.sampling import stratified_sampling
from ..iSIM.sigma import stratified_sigma

class LibChem:
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
            self.smiles = [line.strip() for line in f.readlines()]


    def load_fingerprints(
            self,
            fingerprints : np.array, 
            packed: bool,
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
            sim_index: str = 'JT',
    ) -> np.array:
        """Calculate iSIM similarity matrix from fingerprints"""
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")

        self.iSIM = calculate_isim(unpack_fingerprints(self.fps_packed), n_ary=sim_index)

    def _calculate_iSIM_sigma(
            self,
            n_sigma_samples: int = 50,
            sim_index: str = 'JT',
    ) -> np.array:
        """Calculate iSIM similarity matrix with sigma adjustment from fingerprints"""
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")
        
        self.iSIM_sigma = stratified_sigma(
            unpack_fingerprints(self.fps_packed), 
            n = n_sigma_samples, 
            n_ary = sim_index
        )


    def get_iSIM(
            self,
            sim_index: str = 'JT',
    ) -> np.array:
        """Retrieve the calculated iSIM similarity matrix"""
        if not hasattr(self, 'iSIM'):
            self._calculate_iSIM(sim_index=sim_index)

        return self.iSIM
    
    def get_iSIM_sigma(
            self,
            n_sigma_samples: int = 50,
            sim_index: str = 'JT',
    ) -> float:
        """Retrieve the calculated iSIM sigma value"""
        if not hasattr(self, 'iSIM_sigma'):
            self._calculate_iSIM_sigma(n_sigma_samples=n_sigma_samples, sim_index=sim_index)

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
    
    def cluster(
            self,
            threshold: float = None,
            merge: str = 'diameter',
    ) -> None:
        """Cluster molecules using BitBirch algorithm based on iSIM and optimal threshold"""
        if threshold is None:
            threshold = self._optimal_threshold()
        
        bb_object = BitBirch(threshold=threshold, branching_factor=50, merge_criterion=merge)
        bb_object.fit(self.fps_packed)
        if not hasattr(self, 'iSIM_sigma'):
            self._calculate_iSIM_sigma()

        bb_object.recluster_inplace(extra_threshold=self.iSIM_sigma, verbose=True)
        self.clusters = bb_object.get_cluster_mol_ids()

    def get_clusters(
            self,
    ) -> list[list[int]]:
        """Retrieve the clusters formed after clustering"""
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
    ) -> list[str]:
        """Retrieve the medoid SMILES string for each cluster"""
        if self.smiles is None:
            raise ValueError("SMILES data not loaded.")
        if not hasattr(self, 'clusters'):
            self.cluster()
        
        fingerprints_medoids = []
        smiles_medoids = []
        for cluster in self.clusters:
            fps_cluster = self.fps_packed[cluster]
            fps_cluster = unpack_fingerprints(fps_cluster)
            medoid_index = calculate_medoid(fps_cluster)
            fingerprints_medoids.append(fps_cluster[medoid_index])
            smiles_medoids.append(self.smiles[cluster[medoid_index]])

        return fingerprints_medoids, smiles_medoids
    
    def cluster_sample(
            self,
            n_samples: int = 1000
        ) -> None:
        """Cluster a sample of molecules using stratified sampling"""
        if self.fps_packed is None:
            raise ValueError("Fingerprints not loaded or generated.")
        if self.n_molecules < n_samples:
            raise ValueError("Number of samples exceeds number of molecules.")
        if not hasattr(self, 'clusters'):
            self.cluster()
        if len(self.clusters) > n_samples:
            raise ValueError("Number of clusters is greater than number of samples.")
        if len(self.clusters) == n_samples:
            return self.get_cluster_medoids()
        
        sampled_fps = []
        sampled_smiles = []
        cluster_ids = []
        for k, cluster in enumerate(self.clusters):
            n_cluster = len(cluster)
            fps_cluster = self.fps_packed[cluster]
            fps_cluster = unpack_fingerprints(fps_cluster)
            n_sample_cluster = max(1, int(n_cluster * n_samples / self.n_molecules))
            if n_cluster <= 2:
                sampled_indices = [0]
            else:
                sampled_indices = stratified_sampling(fps_cluster, n_sample=n_sample_cluster)
            sampled_fps_ = fps_cluster[sampled_indices]
            sampled_smiles_ = [self.smiles[cluster[i]] for i in sampled_indices]
            sampled_fps.extend(sampled_fps_)
            sampled_smiles.extend(sampled_smiles_)
            cluster_ids.extend([k] * n_sample_cluster)

        return sampled_fps, sampled_smiles, cluster_ids

