import multiprocessing as mp
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pickle
import argparse
import numpy as np
from iChem.utils import binary_fps, load_smiles
from iChem.bblean import BitBirch, optimal_threshold
import time

def process_and_cluster_chunk(
    chunk_id: int,
    start_index: int,
    smiles_chunk: List[str],
    fp_type: str,
    n_bits: int,
    threshold: Optional[float],
    save_fps: bool,
    output_parent_dir: str,
) -> Tuple[int, float, Dict[str, List[np.ndarray]], Dict[str, List[List[int]]]]:
    """
    Generate and cluster fingerprints for one chunk of SMILES.

    Returns:
        (chunk_id, threshold_used, bfs_dict, mol_indices_dict)
    """
    fps, invalid_smiles = binary_fps(
        smiles_chunk,
        fp_type=fp_type,
        n_bits=n_bits,
        return_invalid=True,
        packed=True,
    )

    if invalid_smiles:
        raise ValueError(f"Chunk {chunk_id} contains invalid SMILES: {invalid_smiles}")

    # Optional: persist packed fingerprint chunk immediately after generation.
    if save_fps:
        chunk_path = Path(output_parent_dir) / f"fps_chunk_{chunk_id:06d}.npy"
        np.save(chunk_path, fps)
    
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
    smi_file: Path,
    fp_type: str = "AP",
    n_bits: int = 2048,
    chunk_size: int = 1_000_000,
    n_workers: Optional[int] = None,
    threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
    save_fps: bool = False,
) -> List[List[int]]:
    """
    Parallelize chunk fingerprint generation and clustering for a large SMILES file,
    then merge chunk-level clusters into final cluster ids.
    
    Args:
        smi_file: Path to input .smi file
        fp_type: Fingerprint type (e.g., 'ecfp4', 'maccs')
        n_bits: Number of bits for fingerprints
        chunk_size: SMILES per worker (default 1M)
        n_workers: Number of parallel workers
        threshold: Clustering threshold
        output_dir: Output path. If it has a file suffix, results are saved there;
            otherwise it is treated as a directory and saved as
            "final_cluster_ids.pkl" inside it.
        save_fps: If True, each worker saves its packed fingerprint chunk right
            after generation in the same directory as the final output file.

    Returns:
        List of clusters, each containing molecule indices.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    basename = smi_file.stem

    if output_dir is None:
        output_path = Path(f"./clustering_{basename}/final_cluster_ids.pkl")
    else:
        output_path = Path(output_dir)

    if output_path.suffix:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "final_cluster_ids.pkl"

    output_parent_dir = str(output_path.parent)
    
    # Read SMILES file in chunks
    print(f"Reading SMILES from {smi_file}...")
    smiles_list = load_smiles(smi_file)
    
    # Split into chunks
    chunks = [
        smiles_list[i:i + chunk_size]
        for i in range(0, len(smiles_list), chunk_size)
    ]
    
    print(f"Processing {len(chunks)} chunks with {n_workers} workers...")
    
    # Build per-chunk tasks with correct global start indices.
    tasks = []
    offset = 0
    for chunk_id, chunk in enumerate(chunks):
        tasks.append(
            (
                chunk_id,
                offset,
                chunk,
                fp_type,
                n_bits,
                threshold,
                save_fps,
                output_parent_dir,
            )
        )
        offset += len(chunk)

    # Generate + cluster each chunk in a single worker call.
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
    parser = argparse.ArgumentParser(description="Parallel chunked clustering for SMILES files")
    parser.add_argument(
        "--smi-file",
        required=True,
        type=Path,
        help="Path to input .smi file",
    )
    args = parser.parse_args()

    smi_file = args.smi_file
    start = time.time()


    # Please change the parameters below as needed for your specific use case. The current settings are just examples.
    parallelize_clustering(
        smi_file=smi_file,
        fp_type="ECFP4",
        n_bits=2048,
        chunk_size=500_000,
        save_fps=True
    )

    end = time.time()
    print(f"Total clustering time: {end - start:.2f} seconds")