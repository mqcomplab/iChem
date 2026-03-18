import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from typing import List, Optional, Tuple, Dict
from iChem.bblean import BitBirch, optimal_threshold
from iChem.bblean.similarity import jt_isim_medoid
from ..utils import load_smiles as _load_smiles
from ..utils import binary_fps

def process_and_cluster_chunk(
    chunk_id: int,
    start_index: int,
    fp_chunk_file: Path,
    threshold: Optional[float],
) -> Tuple[int, float, Dict[str, List[np.ndarray]], Dict[str, List[List[int]]]]:
    """
    Load and cluster one fingerprint chunk stored in a .npy file.

    Returns:
        (chunk_id, threshold_used, bfs_dict, mol_indices_dict)
    """
    fps = np.load(fp_chunk_file)
    
    if threshold is None:
        threshold = optimal_threshold(fps)

    bb_object = BitBirch(
        merge_criterion='diameter',
        threshold=threshold,
    )

    reinsert_ids = np.arange(start_index, start_index + fps.shape[0])
    bb_object.fit(fps, reinsert_indices=reinsert_ids)

    bb_object.delete_internal_nodes()
    fps_bfs, mols_bfs = bb_object._bf_to_np()

    return chunk_id, threshold, fps_bfs, mols_bfs


def process_and_cluster_chunk_array(
    chunk_id: int,
    start_index: int,
    fps: np.ndarray,
    threshold: Optional[float],
) -> Tuple[int, float, Dict[str, List[np.ndarray]], Dict[str, List[List[int]]]]:
    """Cluster one in-memory fingerprint chunk and return BF summaries."""
    if threshold is None:
        threshold = optimal_threshold(fps)

    bb_object = BitBirch(
        merge_criterion='diameter',
        threshold=threshold,
        branching_factor=1024,
    )

    reinsert_ids = np.arange(start_index, start_index + fps.shape[0])
    bb_object.fit(fps, reinsert_indices=reinsert_ids)

    bb_object.delete_internal_nodes()
    fps_bfs, mols_bfs = bb_object._bf_to_np()

    return chunk_id, threshold, fps_bfs, mols_bfs

class LibChemBig:
    def __init__(self,
                 chunk_size: int = 1_000_000,
                 n_workers: int = None,
                 fp_type: str = 'ECFP4',
                 n_bits: int = 2048,
                 threshold: float = None,
                 library_name: Optional[str] = "Lib1Big"
                 ):
        self.chunk_size = chunk_size
        self.n_workers = n_workers or cpu_count()
        self.full_smiles = []
        self.full_fps = None
        self.medoids = []
        self.medoid_fps = []
        self.medoid_smiles = []
        self.fp_type = fp_type
        self.n_bits = n_bits
        self.threshold = threshold
        self.library_name = library_name

    def _load_smiles_gen_fps_indexed(self, task: Tuple[int, str]) -> Tuple[int, List[str], np.ndarray]:
        """Process one file with indexing and return file_index, smiles, and fps for stable merge."""
        file_index, file_path = task
        # Load SMILES and gen fingerprints
        smiles = _load_smiles(file_path)
        fps, _invalid = binary_fps(smiles,
                         fp_type=self.fp_type,
                         n_bits=self.n_bits,
                         packed=True,
                         return_invalid=True)
        
        if len(_invalid) > 0:
            print(f"Warning: {len(_invalid)} invalid SMILES were skipped and deleted.")
            smiles = np.delete(smiles, _invalid, axis=0).tolist()
        
        return file_index, list(smiles), fps
    

    def gen_fps(self, smi_path: str) -> None:
        """Generate fingerprints for all SMILES in a directory of SMI files"""
        
        # Get workers to work separately on each SMI file, then combine results
        smi_files = [os.path.join(smi_path, f) for f in os.listdir(smi_path) if f.endswith('.smi')]
        smi_files.sort()  # Ensure consistent order across runs
        indexed_tasks = [(i, f) for i, f in enumerate(smi_files)]

        # Load the smile file and generate the fingerprints
        # Store the smiles in the corresponding order to be able to retrieve them later for medoid saving
        with Pool(self.n_workers) as pool:
            worker_results = pool.map(self._load_smiles_gen_fps_indexed, indexed_tasks)

        # Keep a deterministic global order regardless of worker completion order.
        worker_results.sort(key=lambda x: x[0])

        full_smiles: List[str] = []
        # Precompute n_cols and dtype from known parameters (packed format: n_bits bits = n_bits//8 bytes)
        n_cols = self.n_bits // 8
        dtype = np.uint8
        total_rows = sum(int(fps_chunk.shape[0]) for _, _, fps_chunk in worker_results)
        full_fps = np.empty((total_rows, n_cols), dtype=dtype) if worker_results else np.empty((0, 0), dtype=dtype)

        offset = 0
        for _, smiles_chunk, fps_chunk in worker_results:
            if len(smiles_chunk) != int(fps_chunk.shape[0]):
                raise ValueError("Mismatch between valid SMILES count and fingerprint rows in chunk.")
            full_smiles.extend(smiles_chunk)
            n_rows = int(fps_chunk.shape[0])
            full_fps[offset:offset + n_rows] = fps_chunk

            offset += n_rows

        self.full_smiles = full_smiles
        self.full_fps = full_fps

    def cluster(self) -> List[List[int]]:
        """Cluster fingerprints in chunks using global-offset reinsert indices."""
        if self.full_fps is None:
            raise ValueError("Fingerprints not generated. Run gen_fps first.")

        n_rows = int(self.full_fps.shape[0])
        if n_rows == 0:
            self.clusters = []
            return self.clusters

        tasks = []
        for chunk_id, start in enumerate(range(0, n_rows, self.chunk_size)):
            fps_chunk = self.full_fps[start:start + self.chunk_size]
            tasks.append((chunk_id, start, fps_chunk, self.threshold))

        with Pool(self.n_workers) as pool:
            bfs_results = pool.starmap(process_and_cluster_chunk_array, tasks)

        final_threshold = self.threshold if self.threshold is not None else max([t[1] for t in bfs_results])

        bbmodel = BitBirch(
            threshold=final_threshold,
            branching_factor=1024,
            merge_criterion='diameter',
        )

        for _, _, bfs_chunk, mols_chunk in bfs_results:
            for key in bfs_chunk.keys():
                bbmodel._fit_np(X=bfs_chunk[key], reinsert_index_seqs=mols_chunk[key])

        self.threshold = final_threshold
        self.clusters = bbmodel.get_cluster_mol_ids()
        return self.clusters


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
            fps_cluster = self.full_fps[cluster]
            medoid_index, medoid_fingerprint = jt_isim_medoid(fps_cluster)
            fingerprints_medoids.append(medoid_fingerprint)
            smiles_medoids.append(self.smiles[cluster[medoid_index]])
        
        self.cluster_medoids_fps = np.array(fingerprints_medoids)
        self.cluster_medoids_smiles = smiles_medoids

        # Save to disk to a .npy and .txt file respectively
        np.save(f'{self.library_name}_medoids_fps.npy', self.cluster_medoids_fps)
        with open(f'{self.library_name}_medoids_smiles.smi', 'w') as f:
            for smi in self.cluster_medoids_smiles:
                f.write(f"{smi}\n")


    def load_medoids_fps_and_smiles(self, fps_file: str, smiles_file: str) -> None:
        """Load medoid fingerprints and SMILES from disk."""
        self.cluster_medoids_fps = np.load(fps_file, mmap_mode='r')
        with open(smiles_file, 'r') as f:
            self.cluster_medoids_smiles = _load_smiles(smiles_file)

    def dump_statistics(self) -> Dict[str, any]:
        """Return a dictionary of statistics about the library and clustering."""
        stats = {
            'library_name': self.library_name,
            'num_molecules': len(self.full_smiles),
            'num_clusters': len(self.clusters) if hasattr(self, 'clusters') else None,
            'threshold': self.threshold,
        }
        return stats


        
       
