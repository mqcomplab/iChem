import multiprocessing as mp
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pickle
import argparse
import numpy as np
from iChem.bblean import BitBirch, optimal_threshold
import time

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


def parallelize_clustering(
    fp_chunks_path: Path,
    n_workers: Optional[int] = None,
    threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Parallelize clustering for chunked fingerprint .npy files,
    then merge chunk-level clusters into final cluster ids.
    
    Args:
        fp_chunks_path: Path to a directory containing chunked .npy fingerprint
            files, or a single .npy file.
        n_workers: Number of parallel workers
        threshold: Clustering threshold
        output_dir: Output path. If it has a file suffix, results are saved there;
            otherwise it is treated as a directory and saved as
            "final_cluster_ids.pkl" inside it.

    Returns:
        List of clusters, each containing molecule indices.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if fp_chunks_path.is_dir():
        chunk_files = sorted(fp_chunks_path.glob("*.npy"))
    elif fp_chunks_path.is_file() and fp_chunks_path.suffix == ".npy":
        chunk_files = [fp_chunks_path]
    else:
        raise ValueError(
            "fp_chunks_path must be a directory with .npy files or a single .npy file"
        )

    if not chunk_files:
        raise ValueError(f"No .npy fingerprint chunk files found in {fp_chunks_path}")

    basename = fp_chunks_path.stem if fp_chunks_path.is_file() else fp_chunks_path.name

    if output_dir is None:
        output_path = Path(f"./clustering_{basename}/final_cluster_ids.pkl")
    else:
        output_path = Path(output_dir)

    if output_path.suffix:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "final_cluster_ids.pkl"

    print(f"Found {len(chunk_files)} fingerprint chunks at {fp_chunks_path}")
    print(f"Processing chunks with {n_workers} workers...")
    
    # Build per-chunk tasks with correct global start indices.
    tasks = []
    offset = 0
    for chunk_id, chunk_file in enumerate(chunk_files):
        chunk_rows = int(np.load(chunk_file, mmap_mode="r").shape[0])
        tasks.append(
            (
                chunk_id,
                offset,
                chunk_file,
                threshold,
            )
        )
        offset += chunk_rows

    # Cluster each chunk in parallel.
    with mp.Pool(n_workers) as pool:
        bf_clusters = pool.starmap(process_and_cluster_chunk, tasks)

    # Cluster the BFS results from all chunks together once they are all done.
    # Create a BitBirch model with the highest of the thresholds used across chunks (or the provided threshold).
    if threshold is None:
        threshold = max(t[1] for t in bf_clusters)

    bbmodel = BitBirch(threshold=threshold, branching_factor=1024, merge_criterion='diameter')

    for chunk_id, _, bfs_chunk, mols_chunk in bf_clusters:
        for key in bfs_chunk.keys():
            bf_to_fit = bfs_chunk[key]
            mols_to_fit = mols_chunk[key]

            bbmodel._fit_np(X=bf_to_fit, reinsert_index_seqs=mols_to_fit)

    # Save and return final cluster ids for all molecules.
    final_cluster_ids = bbmodel.get_cluster_mol_ids()

    with output_path.open("wb") as f:
        pickle.dump(final_cluster_ids, f)
    print(f"Final cluster ids saved to {output_path}")
    print(f"Final number of clusters: {len(final_cluster_ids)}")

    return final_cluster_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel chunked clustering for fingerprint .npy files")
    parser.add_argument(
        "--fps_dir",
        required=True,
        type=Path,
        help="Path to a directory with chunked .npy fingerprints or a single .npy file",
    )
    args = parser.parse_args()

    fp_chunks_path = args.fps_dir
    start = time.time()


    # Please change the parameters below as needed for your specific use case. The current settings are just examples.
    parallelize_clustering(
        fp_chunks_path=fp_chunks_path,
    )

    end = time.time()
    print(f"Total clustering time: {end - start:.2f} seconds")