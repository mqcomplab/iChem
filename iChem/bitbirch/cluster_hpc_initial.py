import gzip as gz
from pathlib import Path

import numpy as np

from bblean import BitBirch #type: ignore
from bblean.multiround import _save_bufs_and_mols #type: ignore
from bblean.fingerprints import _get_fps_file_num #type: ignore

from ..utils import binary_fps, load_smiles


def cluster_hpc_initial(file_path: str,
                        threshold: float,
                        branching_factor: int,
                        merging_criterion: str = 'diameter',
                        reclustering_iterations: int = 3,
                        extra_threshold: float = 0.025,
                        fp_type: str = 'ECFP4',
                        n_bits: int = 2048,
                        output_dir: str = None,
                        start_idx: int = 0,
                        end_idx: int = None,
                        job_label: int = None,
                        ):
    """Initial clustering using BitBirch algorithm for hpc.
    The intent is to submit one job per chunk of data to make it faster and
    more efficient.
    
    Parameters
    ----------
    file_path : str
        Path to the input file containing the data to be clustered.
        File format can be .smi or .smi.gz
    threshold : float
        The threshold for the BitBirch algorithm.
    branching_factor : int
        The branching factor for the BitBirch algorithm.
    merging_criterion : str
        The criterion for merging clusters in the BitBirch algorithm.
    reclustering_iterations : int
        The number of iterations for reclustering in the BitBirch algorithm.
    extra_threshold : float
        The extra threshold for the BitBirch algorithm.
    fp_type : str
        The type of fingerprint to use.
    n_bits : int
        The number of bits in the fingerprint.
    output_dir : str, optional
        The directory where the output files will be saved.

    Returns
    -------
    None
        This function does not return anything.
        It performs the clustering and saves the results to disk.
    """
    file_path = Path(file_path)
    is_smi_gz = file_path.name.endswith('.smi.gz')
    base_name = file_path.name[:-7] if is_smi_gz else file_path.stem
    idx_range = range(start_idx, end_idx)

    # Check the input file
    if not file_path.exists():
        raise ValueError(f"Input file {file_path} does not exist.")
    if file_path.suffix not in ['.smi', '.npy'] and not is_smi_gz:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. Only .smi, "
            f"and .smi.gz are supported."
        )
    # Read the .smi
    if file_path.suffix == '.smi':
        smiles = load_smiles(file_path)
    elif is_smi_gz:
        with gz.open(file_path, 'rt') as f:
            smiles = load_smiles(f)
    fps, invalid_ids = binary_fps(smiles,
                                    fp_type=fp_type,
                                    n_bits=n_bits,
                                    packed=True,
                                    return_invalid=True
                                    )
    if invalid_ids:
        # Drop the invalid ids from the range
        idx_range = [idx for i, idx in enumerate(idx_range) if i not in invalid_ids]
    # Save the fps to a temporary .npy file for clustering
    npy_path = f"{base_name}_fps.npy"
    np.save(npy_path, fps)
    print(f"Saved the fingerprints to {npy_path} for clustering.")

    # Delete fps and smiles from memory to save space
    del fps
    del smiles
   
    # Create the BitBirch instance
    bbobject = BitBirch(threshold=threshold,
                        branching_factor=branching_factor,
                        merging_criterion=merging_criterion
    )

    # Check number of idxs matches the number of rows in the .npy file
    if len(idx_range) != _get_fps_file_num(npy_path):
        raise ValueError(
            f"Number of indices in idx_range ({len(idx_range)}) does not "
            f"match the number of rows in the .npy file ({_get_fps_file_num(npy_path)})."
        )

    # Fit the model to the data
    print(f"Clustering the data in {npy_path} using BitBirch algorithm.")
    bbobject.fit(npy_path, reinsert_indices=idx_range)

    # Remove the temporary .npy file
    try:
        Path(npy_path).unlink()
        print(f"Deleted the temporary file {npy_path}.")
    except Exception as e:
        print(
            f"Warning: Could not delete the temporary file {npy_path}. "
            f"Error: {e}"
        )

    # Recluster if indicated
    if reclustering_iterations > 0:
        bbobject.recluster_inplace(iterations=reclustering_iterations,
                                   extra_threshold=extra_threshold
                                   )
    
    # Save the bufs and the cluster labels to disk
    output_dir = output_dir if output_dir is not None else file_path.parent
    fps_bfs, mols_bfs = bbobject._bf_to_np()
    _save_bufs_and_mols(
        output_dir,
        fps_bfs,
        mols_bfs,
        job_label,
        1
    )