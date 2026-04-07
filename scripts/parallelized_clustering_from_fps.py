import multiprocessing as mp
from pathlib import Path
from typing import List, Optional
import pickle
import argparse
import numpy as np # type: ignore
import time
import glob

from iChem.bblean import BitBirch, optimal_threshold

def process_and_cluster_chunk(
    fps_file: Path,
    chunk_id: int,
    start_index: int,
    threshold: Optional[float],
    output_dir_chunks: Optional[Path] = None,
) -> float:
    """
    Load .npy fingerprint file, cluster with BitBirch, and return BFS results for a single chunk.
    """
    # Load
    fps = np.load(fps_file, mmap_mode='r')
    
    # Get the range of fps indices
    idxs = np.arange(len(fps)) + start_index  # Shift to global indices

    if len(fps) == 0:
        raise ValueError(
            f"Chunk {chunk_id} has no valid fingerprints: {fps_file}"
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
    fps_dir_path: Path,
    n_workers: Optional[int] = None,
    threshold: Optional[float] = None
) -> None:
    """
    Parallelize clustering for chunked fingerprint .npy files,
    then merge chunk-level clusters into final cluster ids.
    
    Args:
        fps_dir_path: Path to a directory containing chunked .npy fingerprint files, or a single .npy file.
        n_workers: Number of parallel workers
        threshold: Clustering threshold

    Returns:
        None
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if fps_dir_path.is_dir():
        chunk_files = sorted(fps_dir_path.glob("*.npy"))
    elif fps_dir_path.is_file() and fps_dir_path.suffix == ".npy":
        chunk_files = [fps_dir_path]
    else:
        raise ValueError(
            "fps_dir_path must be a directory with .npy files or a single .npy file"
        )

    if not chunk_files:
        raise ValueError(f"No .npy fingerprint chunk files found in {fps_dir_path}")

    output_dir_clustering = fps_dir_path / "clustering"
    output_dir_clustering.mkdir(parents=True, exist_ok=True)

    # Make a tmp directory for intermediate chunk results
    output_dir_chunks = output_dir_clustering / "chunks"
    output_dir_chunks.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(chunk_files)} fingerprint chunks at {fps_dir_path}")
    print(f"Processing chunks with {n_workers} workers...")
    
    # Build per-chunk tasks with correct global start indices.
    tasks = []
    offset = 0
    for chunk_id, chunk_file in enumerate(chunk_files):
        # Get chunk size from header only (avoids loading full array into memory)
        with open(chunk_file, 'rb') as f:
            major, _ = np.lib.format.read_magic(f)
            if major == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif major in (2, 3):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(
                    f"Unsupported .npy format version {major}.x for file: {chunk_file}"
                )

        if not shape:
            raise ValueError(f"Invalid fingerprint array shape in file: {chunk_file}")

        chunk_offset = shape[0]  # Number of rows (fingerprints) in this chunk
        tasks.append(
            (
                chunk_file,
                chunk_id,
                offset,
                threshold,
                output_dir_chunks
            )
        )
        offset += chunk_offset

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

    # Recluster the merged model to clean up singletons and refine clusters
    bbmodel.recluster_inplace(iterations=3, extra_threshold=0.025)

    # Save and return final cluster ids for all molecules.
    final_bfs, final_mol_ids = bbmodel._bf_to_np()

    # Delete the intermediate chunk-level BFS results and the directory inclusively
    for chunk_file in chunk_result_files:
        Path(chunk_file).unlink()
    output_dir_chunks.rmdir()

    final_output_path = output_dir_clustering / "final_cluster_bfs.pkl"
    with final_output_path.open("wb") as f:
        pickle.dump((final_bfs, final_mol_ids), f)
    print(f"--------------------------------------------------------------------------------")
    print("***********************************************************************************")
    print(f"Final cluster ids saved to {final_output_path}")
    print(f"Final number of clusters: {np.sum([len(final_bfs[key]) for key in final_bfs.keys()])}")
    print(f"Number of singletons in final clusters: {np.sum([np.sum([len(mol_ids_cluster) == 1 for mol_ids_cluster in final_mol_ids[key]]) for key in final_mol_ids.keys()])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel version to load fps (from .npy) and cluster them")
    parser.add_argument(
        "-fps",
        "--fps_dir",
        required=True,
        type=Path,
        help="Path to a directory with chunked .npy files or a single .npy file",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=None,
        type=float,
        help="Threshold for clustering"
    )
    args = parser.parse_args()

    fps_dir = args.fps_dir
    threshold = args.threshold
    start = time.time()


    # Please change the parameters below as needed for your specific use case. The current settings are just examples.
    parallelize_clustering(
        fps_dir_path=fps_dir,
        threshold=threshold,
    )

    end = time.time()
    print(f"Total clustering time: {end - start:.2f} seconds")