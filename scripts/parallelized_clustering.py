import multiprocessing as mp
from pathlib import Path
from typing import List, Optional
import pickle
import argparse
import numpy as np # type: ignore
import time
import glob


from iChem.bblean import BitBirch, optimal_threshold
from iChem.utils import binary_fps, load_smiles

def process_and_cluster_chunk(
    smi_file: Path,
    chunk_id: int,
    start_index: int,
    threshold: Optional[float],
    output_dir_chunks: Optional[Path] = None,
) -> float:
    """
    Load .smi, generate fps, cluster with BitBirch, and return BFS results for a single chunk.
    """
    # Load smiles
    smiles = load_smiles(smi_file)

    #Generate fps
    fps, invalids = binary_fps(smiles=smiles,
                     fp_type="ECFP4",
                     n_bits=1024,
                     return_invalid=True,
                     packed=True)
    
    # Get the range of fps indices
    idx_range = range(0, len(smiles), 1)

    # Delete invalid indices from the idx_range
    idxs = [idx for idx in idx_range if idx not in invalids]
    idxs = np.array(idxs) + start_index  # Shift to global indices

    if len(fps) == 0:
        raise ValueError(
            f"Chunk {chunk_id} has no valid SMILES after fingerprinting: {smi_file}"
        )

    # Estimate threshold if not provided
    if threshold is None:
        threshold = optimal_threshold(fps)
        print(f"Estimated optimal threshold for chunk {chunk_id}: {threshold:.4f}")

    # Create a bitbirch object and cluster the fps for this chunk
    bb_object = BitBirch(
        merge_criterion='diameter',
        threshold=threshold,
        branching_factor=1024,
    )

    # Cluster the fps for this chunk
    bb_object.fit(fps, reinsert_indices=idxs)

    # Recluster in place to clean up singletons
    bb_object.recluster_inplace(iterations=3, extra_threshold=0.025)

    # Generate BFS results for this chunk
    bb_object.delete_internal_nodes()
    fps_bfs, mols_bfs = bb_object._bf_to_np()

    # Save the chunk-level BFS results to disk for later merging
    if output_dir_chunks:
        chunk_output_path = output_dir_chunks / f"chunk_{chunk_id}_bfs.pkl"
        with chunk_output_path.open("wb") as f:
            pickle.dump((fps_bfs, mols_bfs), f)
        print(f"Chunk {chunk_id} BFS results saved to {chunk_output_path}")

    return threshold

def parallelize_clustering(
    smiles_dir_path: Path,
    n_workers: Optional[int] = None,
    threshold: Optional[float] = None,
) -> None:
    """
    Parallelize clustering for chunked fingerprint .npy files,
    then merge chunk-level clusters into final cluster ids.
    
    Args:
        smiles_dir_path: Path to a directory containing chunked .smi files, or a single .smi file.
        n_workers: Number of parallel workers
        threshold: Clustering threshold

    Returns:
        None
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if smiles_dir_path.is_dir():
        chunk_files = sorted(smiles_dir_path.glob("*.smi"))
    elif smiles_dir_path.is_file() and smiles_dir_path.suffix == ".smi":
        chunk_files = [smiles_dir_path]
    else:
        raise ValueError(
            "smiles_dir_path must be a directory with .smi files or a single .smi file"
        )

    if not chunk_files:
        raise ValueError(f"No .smi fingerprint chunk files found in {smiles_dir_path}")

    output_dir_clustering = smiles_dir_path / "clustering"
    output_dir_clustering.mkdir(parents=True, exist_ok=True)

    # Make a tmp directory for intermediate chunk results
    output_dir_chunks = output_dir_clustering / "chunks"
    output_dir_chunks.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(chunk_files)} fingerprint chunks at {smiles_dir_path}")
    print(f"Processing chunks with {n_workers} workers...")
    
    # Build per-chunk tasks with correct global start indices.
    tasks = []
    offset = 0
    for chunk_id, chunk_file in enumerate(chunk_files):
        chunk_rows = sum(1 for _ in chunk_file.open())  # Count lines in the .smi file
        tasks.append(
            (
                chunk_file,
                chunk_id,
                offset,
                threshold,
                output_dir_chunks
            )
        )
        offset += chunk_rows

    # Cluster each chunk in parallel.
    with mp.Pool(n_workers) as pool:
        thresholds_chunks = pool.starmap(process_and_cluster_chunk, tasks)

    if threshold is None:
        threshold = np.max(thresholds_chunks)
        print(f"Using maximum estimated threshold across chunks for merging: {threshold:.4f}")

    # Create a new BitBirch model to merge chunk-level clusters.
    bbmodel = BitBirch(threshold=threshold,
                       branching_factor=1024,
                       merge_criterion='diameter')
    
    # Load all chunk-level BFS results and fit them into the new BitBirch model for merging.
    chunk_result_files = sorted(glob.glob(str(output_dir_chunks / "chunk_*_bfs.pkl")))
    for chunk_file in chunk_result_files:
        with open(chunk_file, "rb") as f:
            bfs, mol_ids = pickle.load(f)
        for key in bfs.keys():
            bf_to_fit = bfs[key]
            mols_to_fit = mol_ids[key]
            bbmodel._fit_np(X=bf_to_fit, reinsert_index_seqs=mols_to_fit)

    # Save and return final cluster ids for all molecules.
    final_bfs, final_mol_ids = bbmodel._bf_to_np()

    # Delete the intermediate chunk-level BFS results and the directory inclusively
    for chunk_file in chunk_result_files:
        Path(chunk_file).unlink()
    output_dir_chunks.rmdir()

    final_output_path = output_dir_clustering / "final_cluster_ids.pkl"
    with final_output_path.open("wb") as f:
        pickle.dump((final_bfs, final_mol_ids), f)
    print(f"Final cluster ids saved to {final_output_path}")
    print(f"Final number of clusters: {len(final_bfs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel version to generate fps (from .smi) and cluster them")
    parser.add_argument(
        "--smiles_dir",
        required=True,
        type=Path,
        help="Path to a directory with chunked .smi files or a single .smi file",
    )
    args = parser.parse_args()

    smiles_dir = args.smiles_dir
    start = time.time()


    # Please change the parameters below as needed for your specific use case. The current settings are just examples.
    parallelize_clustering(
        smiles_dir_path=smiles_dir,
    )

    end = time.time()
    print(f"Total clustering time: {end - start:.2f} seconds")