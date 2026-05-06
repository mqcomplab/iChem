import gzip as gz
from pathlib import Path

import numpy as np

from bblean.bitbirch import BitBirch # type: ignore

from .optimal_threshold import optimal_threshold
from ..utils import binary_fps, load_smiles

def cluster(file_path: str,
            threshold: float = None,
            fp_type: str = 'ECFP4',
            n_bits: int = 2048,
            branching_factor: int = 1024,
            merge_criterion: str = 'diameter',
            recluster_iterations: int = 5,
            recluster_extra_threshold: float = 0.025,
            verbose: bool = False):
    """Cluster the molecules using the best_practices recommendations on the
    paper.

    Parameters
    ----------
    file_path : Path
        Path to the file containing the .npy or .smi data.
    threshold : float, optional
        Similarity threshold for clustering. If None, it will be determined
        automatically.
    fp_type : str
        Type of fingerprints to use. 'ECFP4' or 'ECFP6', 'RDKIT', 'AP' etc.
    n_bits : int
        Number of bits for the fingerprints.
    branching_factor : int
        Branching factor for the BitBirch algorithm.
    merge_criterion : str, optional
        Criterion for merging nodes in the BitBirch algorithm.

    Returns
    -------
    cluster_ids : list
        A list of clusters, where each cluster is a list of molecule IDs. Save
        in a .pkl file.
    """
    file_path = Path(file_path)
    # Check if input is a directory or a file
    if file_path.is_file():
        if file_path.name.endswith(('.smi', '.smi.gz')):
            if verbose:
                print(f"Processing file: {file_path}")
            return _cluster_from_smile_file(
                file_path,
                threshold,
                fp_type,
                n_bits,
                branching_factor,
                merge_criterion,
                verbose,
            )
        elif file_path.suffix == '.npy':
            if verbose:
                print(f"Processing fingerprint file: {file_path}")
            return _cluster_from_npy_file(
                file_path,
                threshold,
                branching_factor,
                merge_criterion,
                recluster_iterations,
                recluster_extra_threshold,
                verbose,
            )
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    elif file_path.is_dir():
        # Read all .npy files in the directory and cluster them sequentially
        npy_files = sorted(file_path.glob('*.npy'))

        # Read all .smi files in the directory and cluster them sequentially
        smi_files = sorted(file_path.glob('*.smi')) + sorted(file_path.glob('*.smi.gz'))

        if len(npy_files) > 0:
            if verbose:
                print(
                    f"Found {len(npy_files)} .npy files in directory. "
                    "Clustering sequentially..."
                )
            return _cluster_from_directory_npy(
                file_path,
                threshold,
                branching_factor,
                merge_criterion,
                recluster_iterations,
                recluster_extra_threshold,
                verbose
            )
        elif len(smi_files) > 0:
            if verbose:
                print(
                    f"Found {len(smi_files)} .smi files in directory. "
                    "Clustering sequentially..."
                )
            return _cluster_from_directory_smile(
                file_path,
                threshold,
                fp_type,
                n_bits,
                branching_factor,
                merge_criterion,
                recluster_iterations,
                recluster_extra_threshold,
                verbose
            )
        else:
            raise ValueError(f"No .npy or .smi files found in directory: {file_path}")
    else:
        raise ValueError(f"Invalid input path: {file_path}")

def _cluster_from_npy_file(file_path: Path,
            threshold: float = None,
            branching_factor: int = 1024,
            merge_criterion: str = 'diameter',
            recluster_iterations: int = 5,
            recluster_extra_threshold: float = 0.025,
            verbose: bool = False):
    """Cluster the molecules from a single .npy file"""
    # Create the BitBirch instance
    bb_object = BitBirch(
        merge_criterion=merge_criterion,
        threshold=threshold,
        branching_factor=branching_factor,
    )
    # Fit the fingerprints into the BitBirch model
    bb_object.fit(file_path)
    # Recluster to decrease the number of clusters
    bb_object.recluster_inplace(
        iterations=recluster_iterations,
        extra_threshold=recluster_extra_threshold,
        verbose=verbose,
    )
    return bb_object.get_cluster_mol_ids()

def _cluster_from_smile_file(file_path: Path,
            threshold: float = None,
            fp_type: str = 'ECFP4',
            n_bits: int = 2048,
            branching_factor: int = 1024,
            merge_criterion: str = 'diameter',
            recluster_iterations: int = 5,
            recluster_extra_threshold: float = 0.025,
            verbose: bool = False):
    """Cluster the molecules from a single .smi file"""
    import gzip as gz

    # Check if the file is gzipped    
    if file_path.suffix == '.gz':
        with gz.open(file_path, 'rt') as f:
            smiles = [line.strip() for line in f]
    else:
        smiles = load_smiles(file_path)

    # Generate the fingerprints
    fps, invalid_ids = binary_fps(
        smiles,
        fp_type=fp_type,
        n_bits=n_bits,
        packed=True,
        return_invalid=True,
    )

    if verbose:
        print(f"Generated fingerprints for {len(fps)} molecules.")

    # Write the fingerprints to a .npy file
    npy_file_path = file_path.with_suffix('.npy')
    np.save(npy_file_path, fps)

    # Drop the invalid smiles and rewrite the .smi
    if len(invalid_ids) > 0:
        if verbose:
            print(f"Warning: {len(invalid_ids)} invalid SMILES were skipped.")
        valid_smiles = [smi for i, smi in enumerate(smiles) if i not in invalid_ids]
        if file_path.suffix == '.gz':
            corrected_file_path = file_path.with_name(
                file_path.name.replace('.smi.gz', '_valid.smi')
            )
        else:
            corrected_file_path = file_path.with_name(file_path.stem + '_valid.smi')

        # Rewrite the .smi file with only valid smiles
        with open(corrected_file_path, 'w') as f:
            for smi in valid_smiles:
                f.write(f"{smi}\n")

    # Determine the optimal threshold if not provided
    if threshold is None:
        if verbose:
            print("Determining optimal threshold...")
        threshold = optimal_threshold(fps, factor=3.5)
        if verbose:
            print(f"Optimal threshold determined: {threshold:.4f}")

    # Create the BitBirch instance
    bb_object = BitBirch(
        merge_criterion=merge_criterion,
        threshold=threshold,
        branching_factor=branching_factor,
    )

    # Fit the fingerprints into the BitBirch model
    bb_object.fit(npy_file_path)

    # Recluster to decrease the number of clusters
    bb_object.recluster_inplace(
        iterations=recluster_iterations,
        extra_threshold=recluster_extra_threshold,
        verbose=verbose,
    )

    # Delete the temporary .npy file
    npy_file_path.unlink()

    return bb_object.get_cluster_mol_ids()

def _cluster_from_directory_npy(dir_path: Path,
            threshold: float = None,
            branching_factor: int = 1024,
            merge_criterion: str = 'diameter',
            recluster_iterations: int = 5,
            recluster_extra_threshold: float = 0.025,
            verbose: bool = False):
    """Cluster the molecules from all .npy files in a directory"""
    print(f"Clustering all .npy files in directory: {dir_path}")
    print(
        "WARNING: This sequential clustering might not be optimal for sets "
        "above 100 million molecules."
    )
    print("Consider using multiround for larger datasets.")

    # Find all .npy files in the directory
    npy_files = sorted(dir_path.glob('*.npy'))

    # Create the BitBirch instance
    bb_object = BitBirch(
        merge_criterion=merge_criterion,
        threshold=threshold,
        branching_factor=branching_factor,
    )

    for npy_file in npy_files:
        if verbose:
            print(f"Processing file: {npy_file}")
        if threshold is None and npy_file == npy_files[0]:
            # Determine threshold only for the first file if not provided.
            if verbose:
                print(f"Determining optimal threshold for file {npy_file}...")
            threshold = optimal_threshold(
                np.load(npy_file, mmap_mode='r'),
                factor=3.5,
            )
            if verbose:
                print(
                    f"Optimal threshold determined for file {npy_file}: "
                    f"{threshold:.4f}"
                )
                print(f"This threshold will be used for all files in the directory.")
                print(
                    "If other threshold wanted, please provide it as an "
                    "argument to the function."
                )
            bb_object.threshold = threshold
        bb_object.fit(npy_file)
    
    # Recluster to decrease the number of clusters
    bb_object.recluster_inplace(
        iterations=recluster_iterations,
        extra_threshold=recluster_extra_threshold,
        verbose=verbose,
    )
    
    return bb_object.get_cluster_mol_ids()

def _cluster_from_directory_smile(dir_path: Path,
            threshold: float = None,
            fp_type: str = 'ECFP4',
            n_bits: int = 2048,
            branching_factor: int = 1024,
            merge_criterion: str = 'diameter',
            recluster_iterations: int = 5,
            recluster_extra_threshold: float = 0.025,
            verbose: bool = False):
    """Cluster the molecules from all .smi files in a directory"""
    print(f"Clustering all .smi files in directory: {dir_path}")
    print(
        "WARNING: This sequential clustering might not be optimal for sets "
        "above 100 million molecules."
    )
    print("Consider using multiround for larger datasets.")

    # Find all .smi files in the directory
    smi_files = sorted(dir_path.glob('*.smi')) + sorted(dir_path.glob('*.smi.gz'))

    # Create the BitBirch instance
    bb_object = BitBirch(
        merge_criterion=merge_criterion,
        threshold=threshold,
        branching_factor=branching_factor,
    )

    for smi_file in smi_files:
        # Check if the file is gzipped
        if smi_file.suffix == '.gz':
            with gz.open(smi_file, 'rt') as f:
                smiles = [line.strip() for line in f]
        else:
            smiles = load_smiles(smi_file)
        temp_fps, invalid_ids = binary_fps(smiles,
                                fp_type=fp_type,
                                n_bits=n_bits,
                                packed=True,
                                return_invalid=True)
        if invalid_ids:
            if verbose:
                print(
                    f"Warning: {len(invalid_ids)} invalid SMILES were skipped in "
                    f"file {smi_file}."
                )
            valid_smiles = [smi for i, smi in enumerate(smiles) if i not in invalid_ids]
            if smi_file.suffix == '.gz':
                corrected_file_path = smi_file.with_name(
                    smi_file.name.replace('.smi.gz', '_valid.smi')
                )
            else:
                corrected_file_path = smi_file.with_name(smi_file.stem + '_valid.smi')

            # Rewrite the .smi file with only valid smiles
            with open(corrected_file_path, 'w') as f:
                for smi in valid_smiles:
                    f.write(f"{smi}\n")
        if threshold is None and smi_file == smi_files[0]:
            # Determine threshold only for the first file if not provided.
            if verbose:
                print(f"Determining optimal threshold for file {smi_file}...")
            threshold = optimal_threshold(temp_fps, factor=3.5)
            if verbose:
                print(
                    f"Optimal threshold determined for file {smi_file}: "
                    f"{threshold:.4f}"
                )
                print(f"This threshold will be used for all files in the directory.")
                print(
                    "If other threshold wanted, please provide it as an "
                    "argument to the function."
                )
            bb_object.threshold = threshold
        temp_npy_file = smi_file.with_suffix('.npy')
        np.save(temp_npy_file, temp_fps)
        bb_object.fit(temp_npy_file)
        temp_npy_file.unlink()

    # Recluster to decrease the number of clusters
    bb_object.recluster_inplace(
        iterations=recluster_iterations,
        extra_threshold=recluster_extra_threshold,
        verbose=verbose,
    )

    return bb_object.get_cluster_mol_ids()