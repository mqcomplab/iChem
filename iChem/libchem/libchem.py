import numpy as np # type: ignore

from ..bblean import pack_fingerprints, unpack_fingerprints
from ..utils import binary_fps
from ..bblean import BitBirch
from ..iSIM import calculate_isim
from ..iSIM.sigma import stratified_sigma

class LibChem:
    def __init__(self):
        self.smiles = None
        self.fps_packed = None

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
        self.iSIM_sigma = self.iSIM_sigma

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
            self._calculate_iSIM_sigma(self)
        if not hasattr(self, 'iSIM'):
            self._calculate_iSIM(self)

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
        self.clusters = bb_object.get_cluster_mol_ids()


