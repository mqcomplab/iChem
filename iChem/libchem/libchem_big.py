import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from typing import List, Optional, Tuple, Dict
from iChem.bblean import BitBirch, optimal_threshold
from iChem.bblean.similarity import jt_isim_medoid
from ..utils import load_smiles as _load_smiles
from ..utils import binary_fps

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

    def _load_smiles_gen_fps_cluster(self, task: Tuple[int, int, str]) -> Tuple[float, List[str], np.ndarray]:
        """Process one file with indexing and return file_index, smiles, and fps for stable merge."""
        initial_id, final_id, file_path = task
        print(f"Processing file: {file_path} with IDs from {initial_id} to {final_id}")

        reinsert_ids = np.arange(initial_id, final_id)

        # Load SMILES and gen fingerprints
        smiles = _load_smiles(file_path)
        fps, _invalid = binary_fps(smiles,
                         fp_type=self.fp_type,
                         n_bits=self.n_bits,
                         packed=True,
                         return_invalid=True)
        
        if len(_invalid) > 0:
            print(f"Warning: {len(_invalid)} invalid SMILES were skipped.")
            # Delete invalid entries from reinsert_ids to maintain correct indexing, note that invalid are the positions in the array not the actual id
            reinsert_ids = np.delete(reinsert_ids, _invalid)

        if self.threshold is None:
            threshold = optimal_threshold(fps)

        bb_object = BitBirch(
        merge_criterion='diameter',
        threshold=threshold,
        branching_factor=1024,
        )

        bb_object.fit(fps, reinsert_indices=reinsert_ids)

        bb_object.delete_internal_nodes()
        fps_bfs, mols_bfs = bb_object._bf_to_np()

        del bb_object
        del fps

        return threshold, fps_bfs, mols_bfs
    

    def _load_fps_cluster(self, task: Tuple[int, int, str]) -> Tuple[float, List[str], np.ndarray]:
        """Process one file with indexing and return file_index, smiles, and fps for stable merge."""
        initial_id, final_id, file_path = task
        print(f"Processing file: {file_path} with IDs from {initial_id} to {final_id}")

        reinsert_ids = np.arange(initial_id, final_id)

        fps = np.load(file_path, mmap_mode='r')
        
        if self.threshold is None:
            threshold = optimal_threshold(fps)

        bb_object = BitBirch(
        merge_criterion='diameter',
        threshold=threshold,
        branching_factor=1024,
        )

        bb_object.fit(fps, reinsert_indices=reinsert_ids)

        bb_object.delete_internal_nodes()
        fps_bfs, mols_bfs = bb_object._bf_to_np()

        del bb_object
        del fps

        return threshold, fps_bfs, mols_bfs
    

    def gen_fps_and_cluster(self, smi_path: str) -> None:
        """Generate fingerprints for all SMILES in a directory of SMI files and cluster them"""

        # Get workers to work separately on each SMI file, then combine results
        smi_files = [os.path.join(smi_path, f) for f in os.listdir(smi_path) if f.endswith('.smi')]
        smi_files.sort()  # Ensure consistent order across runs
    
        # Get the initial and final id for each file based on the chunk size, to be used for reinsert indices in clustering
        indexed_tasks = []
        current_id = 0
        for smi_file in smi_files:
            num_lines = self.chunk_size
            final_id = current_id + num_lines
            indexed_tasks.append((current_id, final_id, smi_file))
            current_id = final_id

        # For the last task, modify the final_id to be the total number of lines in the file, to avoid out of bounds error
        last_file = smi_files[-1]
        with open(last_file, 'r') as f:
            total_lines = sum(1 for _ in f)
        indexed_tasks[-1] = (indexed_tasks[-1][0], indexed_tasks[-1][0] + total_lines, last_file)

        # Load the smile file and generate the fingerprints
        # Store the smiles in the corresponding order to be able to retrieve them later for medoid saving
        with Pool(self.n_workers) as pool:
            worker_results = pool.map(self._load_smiles_gen_fps_cluster, indexed_tasks)

        # Combine results from all workers
        final_threshold = self.threshold if self.threshold is not None else max([t[0] for t in worker_results])

        bbmodel = BitBirch(
            threshold=final_threshold,
            branching_factor=1024,
            merge_criterion='diameter',
        )

        for _, bfs_chunk, mols_chunk in worker_results:
            for key in bfs_chunk.keys():
                bbmodel._fit_np(X=bfs_chunk[key], reinsert_index_seqs=mols_chunk[key])

        centroids_fps = bbmodel.get_centroids()
        del bbmodel
        # Save centroids_fps
        np.save(f'{self.library_name}_centroids_fps.npy', centroids_fps)

        # Print statistics about the library and clustering
        print(f"Number of molecules clustered: {indexed_tasks[-1][0] + total_lines}")
        print(f"Number of clusters formed: {len(centroids_fps)}")
        print(f"Clustering threshold used: {final_threshold:.4f}")

    def load_fps_and_cluster(self, fps_dir: str) -> None:
        """Load precomputed fingerprints and cluster them"""
        # Load the precomputed fingerprints
        fps_files = [os.path.join(fps_dir, f) for f in os.listdir(fps_dir) if f.endswith('.npy')]
        fps_files.sort()  # Ensure consistent order across runs

        # Get the initial and final id for each file based on the chunk size, to be used for reinsert indices in clustering
        indexed_tasks = []
        current_id = 0
        for fps_file in fps_files:
            num_lines = self.chunk_size
            final_id = current_id + num_lines
            indexed_tasks.append((current_id, final_id, fps_file))
            current_id = final_id

        # For the last task, modify the final_id to be the total number of lines in the file, to avoid out of bounds error
        last_file = fps_files[-1]
        fps_last = np.load(last_file, mmap_mode='r')
        total_lines = fps_last.shape[0]
        indexed_tasks[-1] = (indexed_tasks[-1][0], indexed_tasks[-1][0] + total_lines, last_file)

        # Load the precomputed fingerprints
        with Pool(self.n_workers) as pool:
            worker_results = pool.map(self._load_fps_cluster, indexed_tasks)

        # Combine results from all workers
        final_threshold = self.threshold if self.threshold is not None else max([t[0] for t in worker_results])

        bbmodel = BitBirch(
            threshold=final_threshold,
            branching_factor=1024,
            merge_criterion='diameter',
        )

        for _, bfs_chunk, mols_chunk in worker_results:
            for key in bfs_chunk.keys():
                bbmodel._fit_np(X=bfs_chunk[key], reinsert_index_seqs=mols_chunk[key])

        # Save centroids_fps
        centroids_fps = bbmodel.get_centroids()
        del bbmodel
        np.save(f'{self.library_name}_centroids_fps.npy', centroids_fps)

        # Print statistics about the library and clustering
        print(f"Number of molecules clustered: {indexed_tasks[-1][0] + total_lines}")
        print(f"Number of clusters formed: {len(centroids_fps)}")
        print(f"Clustering threshold used: {final_threshold:.4f}")


        
       
