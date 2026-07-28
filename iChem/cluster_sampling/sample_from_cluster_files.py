import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from bblean.similarity import jt_isim_medoid, jt_sim_packed  # type: ignore
from ..utils import load_smiles, load_smiles_gzipped
from ..utils.fingerprints import _binary_fps as binary_fps


def _worker_medoid_sampling_from_file(cluster_idx: int, cluster_file: Path,
                                       fp_type: str, n_bits: int,
                                       sample: bool, sample_min_size: int) -> tuple[int, str]:
    """Worker function to compute medoid for a single cluster file.

    Returns (cluster_idx, sampled_smiles) for order preservation.
    """
    print(f"[Cluster {cluster_idx}] Processing: {cluster_file.name}", flush=True)
    sys.stdout.flush()
    
    # Load SMILES from file
    if cluster_file.suffix == '.smi':
        smiles_list = load_smiles(cluster_file)
    elif cluster_file.suffix == '.gz':
        smiles_list = load_smiles_gzipped(cluster_file)
    else:
        print(f"[Cluster {cluster_idx}] Error: Unsupported file format", flush=True)
        sys.stdout.flush()
        return None

    if len(smiles_list) == 0:
        print(f"[Cluster {cluster_idx}] Error: Empty cluster", flush=True)
        sys.stdout.flush()
        return None

    print(f"[Cluster {cluster_idx}] Loaded {len(smiles_list)} SMILES", flush=True)
    sys.stdout.flush()

    # Sample SMILES first if needed
    if sample and len(smiles_list) > sample_min_size:
        print(f"[Cluster {cluster_idx}] Subsampling to {sample_min_size} molecules", flush=True)
        sys.stdout.flush()
        sample_indices = np.random.choice(len(smiles_list), size=sample_min_size, replace=False)
        sampled_smiles = [smiles_list[i] for i in sample_indices]
    else:
        sampled_smiles = smiles_list
        sample_indices = np.arange(len(smiles_list))

    # Compute fingerprints only on sampled molecules
    print(f"[Cluster {cluster_idx}] Computing fingerprints ({fp_type}, {n_bits} bits)", flush=True)
    sys.stdout.flush()
    sample_fps = binary_fps(sampled_smiles, fp_type=fp_type, n_bits=n_bits, return_invalid=False, packed=True)

    medoid_idx, _ = jt_isim_medoid(sample_fps)
    print(f"[Cluster {cluster_idx}] ✓ Completed", flush=True)
    sys.stdout.flush()
    return (cluster_idx, sampled_smiles[medoid_idx])


def _worker_centroid_like_sampling_from_file(cluster_idx: int, cluster_file: Path,
                                              centroid: np.ndarray,
                                              fp_type: str, n_bits: int,
                                              sample: bool, sample_min_size: int) -> tuple[int, str]:
    """Worker function to compute centroid-like for a single cluster file.

    Returns (cluster_idx, sampled_smiles) for order preservation.
    """
    print(f"[Cluster {cluster_idx}] Processing: {cluster_file.name}", flush=True)
    sys.stdout.flush()
    
    # Load SMILES from file
    if cluster_file.suffix == '.smi':
        smiles_list = load_smiles(cluster_file)
    elif cluster_file.suffix == '.gz':
        smiles_list = load_smiles_gzipped(cluster_file)
    else:
        print(f"[Cluster {cluster_idx}] Error: Unsupported file format", flush=True)
        sys.stdout.flush()
        return None

    if len(smiles_list) == 0:
        print(f"[Cluster {cluster_idx}] Error: Empty cluster", flush=True)
        sys.stdout.flush()
        return None

    print(f"[Cluster {cluster_idx}] Loaded {len(smiles_list)} SMILES", flush=True)
    sys.stdout.flush()

    # Sample SMILES first if needed
    if sample and len(smiles_list) > sample_min_size:
        print(f"[Cluster {cluster_idx}] Subsampling to {sample_min_size} molecules", flush=True)
        sys.stdout.flush()
        sample_indices = np.random.choice(len(smiles_list), size=sample_min_size, replace=False)
        sampled_smiles = [smiles_list[i] for i in sample_indices]
    else:
        sampled_smiles = smiles_list
        sample_indices = np.arange(len(smiles_list))

    # Compute fingerprints only on sampled molecules
    print(f"[Cluster {cluster_idx}] Computing fingerprints ({fp_type}, {n_bits} bits)", flush=True)
    sys.stdout.flush()
    sample_fps = binary_fps(sampled_smiles, fp_type=fp_type, n_bits=n_bits, return_invalid=False, packed=True)

    similarities = jt_sim_packed(sample_fps, centroid)
    closest_idx = np.argmax(similarities)
    print(f"[Cluster {cluster_idx}] ✓ Completed", flush=True)
    sys.stdout.flush()
    return (cluster_idx, sampled_smiles[closest_idx])


def _worker_centroid_stratified_sampling_from_file(cluster_idx: int, cluster_file: Path,
                                                   centroid: np.ndarray,
                                                   fp_type: str, n_bits: int,
                                                   sample: bool, sample_min_size: int) -> tuple[int, str]:
    """Worker function to compute centroid-stratified for a single cluster file.

    Returns (cluster_idx, sampled_smiles) for order preservation.
    """
    print(f"[Cluster {cluster_idx}] Processing: {cluster_file.name}", flush=True)
    sys.stdout.flush()
    
    # Load SMILES from file
    if cluster_file.suffix == '.smi':
        smiles_list = load_smiles(cluster_file)
    elif cluster_file.suffix == '.gz':
        smiles_list = load_smiles_gzipped(cluster_file)
    else:
        print(f"[Cluster {cluster_idx}] Error: Unsupported file format", flush=True)
        sys.stdout.flush()
        return None

    if len(smiles_list) == 0:
        print(f"[Cluster {cluster_idx}] Error: Empty cluster", flush=True)
        sys.stdout.flush()
        return None

    print(f"[Cluster {cluster_idx}] Loaded {len(smiles_list)} SMILES", flush=True)
    sys.stdout.flush()

    # Sample SMILES first if needed
    if sample and len(smiles_list) > sample_min_size:
        print(f"[Cluster {cluster_idx}] Subsampling to {sample_min_size} molecules", flush=True)
        sys.stdout.flush()
        sample_indices = np.random.choice(len(smiles_list), size=sample_min_size, replace=False)
        sampled_smiles = [smiles_list[i] for i in sample_indices]
    else:
        sampled_smiles = smiles_list
        sample_indices = np.arange(len(smiles_list))

    # Compute fingerprints only on sampled molecules
    print(f"[Cluster {cluster_idx}] Computing fingerprints ({fp_type}, {n_bits} bits)", flush=True)
    sys.stdout.flush()
    sample_fps = binary_fps(sampled_smiles, fp_type=fp_type, n_bits=n_bits, return_invalid=False, packed=True)

    similarities = jt_sim_packed(sample_fps, centroid)
    # Sample the max
    closest_idx = np.argmax(similarities)
    # Sample the min
    furthest_idx = np.argmin(similarities)
    # Sample the mean
    mean_similarity = np.argmin(np.abs(similarities - np.mean(similarities)))
    print(f"[Cluster {cluster_idx}] Closest: {sampled_smiles[closest_idx]}, Furthest: {sampled_smiles[furthest_idx]}, Mean: {sampled_smiles[mean_similarity]}")
    print(f"[Cluster {cluster_idx}] ✓ Completed", flush=True)
    sys.stdout.flush()
    return [(cluster_idx, sampled_smiles[closest_idx]),
            (cluster_idx, sampled_smiles[furthest_idx]),
            (cluster_idx, sampled_smiles[mean_similarity])]


def _medoids_sampling_from_files(cluster_files: list[Path],
                                  fp_type: str = 'ECFP4',
                                  n_bits: int = 2048,
                                  sample: bool = True,
                                  sample_min_size: int = 1_000,
                                  n_processes: int = None) -> list[str]:
    """Sample medoid from each cluster file in parallel."""
    if n_processes is None or n_processes <= 0:
        n_processes = min(8, cpu_count())

    if n_processes == 1:
        # Sequential mode
        results = []
        for idx, cluster_file in enumerate(cluster_files):
            result = _worker_medoid_sampling_from_file(idx, cluster_file, fp_type, n_bits, sample, sample_min_size)
            if result is not None:
                results.append(result)
        results.sort(key=lambda x: x[0])
        return [smiles for _, smiles in results]

    # Parallel mode
    n_processes = min(n_processes, len(cluster_files))
    tasks = [(idx, cluster_file, fp_type, n_bits, sample, sample_min_size)
             for idx, cluster_file in enumerate(cluster_files)]

    show_progress = len(cluster_files) > 10
    with Pool(processes=n_processes) as pool:
        if show_progress:
            results = list(tqdm(pool.starmap(_worker_medoid_sampling_from_file, tasks),
                               total=len(tasks), desc="Sampling medoids from cluster files"))
        else:
            results = pool.starmap(_worker_medoid_sampling_from_file, tasks)

    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])
    return [smiles for _, smiles in results]


def _centroid_like_sampling_from_files(cluster_files: list[Path],
                                        centroids: np.ndarray,
                                        fp_type: str = 'ECFP4',
                                        n_bits: int = 2048,
                                        sample: bool = True,
                                        sample_min_size: int = 1_000,
                                        n_processes: int = None) -> list[str]:
    """Sample centroid-like from each cluster file in parallel."""
    if n_processes is None or n_processes <= 0:
        n_processes = min(8, cpu_count())

    if n_processes == 1:
        # Sequential mode
        results = []
        for idx, cluster_file in enumerate(cluster_files):
            centroid = centroids[idx]
            result = _worker_centroid_like_sampling_from_file(idx, cluster_file, centroid, fp_type, n_bits, sample, sample_min_size)
            if result is not None:
                results.append(result)
        results.sort(key=lambda x: x[0])
        return [smiles for _, smiles in results]

    # Parallel mode
    n_processes = min(n_processes, len(cluster_files))
    tasks = [(idx, cluster_file, centroids[idx], fp_type, n_bits, sample, sample_min_size)
             for idx, cluster_file in enumerate(cluster_files)]

    show_progress = len(cluster_files) > 10
    with Pool(processes=n_processes) as pool:
        if show_progress:
            results = list(tqdm(pool.starmap(_worker_centroid_like_sampling_from_file, tasks),
                               total=len(tasks), desc="Sampling centroid-like from cluster files"))
        else:
            results = pool.starmap(_worker_centroid_like_sampling_from_file, tasks)

    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])
    return [smiles for _, smiles in results]

def _centroid_stratified_sampling_from_files(cluster_files: list[Path],
                                              centroids: np.ndarray,
                                              fp_type: str = 'ECFP4',
                                              n_bits: int = 2048,
                                              sample: bool = True,
                                              sample_min_size: int = 1_000,
                                              n_processes: int = None) -> list[str]:
    """Sample centroid-stratified from each cluster file in parallel."""
    # For centroid-stratified, we use the centroid-stratified sampling
    if n_processes is None or n_processes <= 0:
        n_processes = min(8, cpu_count())
    
    if n_processes == 1:
        # Sequential mode
        results = []
        for idx, cluster_file in enumerate(cluster_files):
            centroid = centroids[idx]
            result = _worker_centroid_stratified_sampling_from_file(idx, cluster_file, centroid, fp_type, n_bits, sample, sample_min_size)
            if result is not None:
                results.extend(result)  # result is a list of tuples
        results.sort(key=lambda x: x[0])
        return [smiles for _, smiles in results]
    
    # Parallel mode
    n_processes = min(n_processes, len(cluster_files))
    tasks = [(idx, cluster_file, centroids[idx], fp_type, n_bits, sample, sample_min_size)
             for idx, cluster_file in enumerate(cluster_files)]

    show_progress = len(cluster_files) > 10
    with Pool(processes=n_processes) as pool:
        if show_progress:
            results = list(tqdm(pool.starmap(_worker_centroid_stratified_sampling_from_file, tasks),
                               total=len(tasks), desc="Sampling centroid-stratified from cluster files"))
        else:
            results = pool.starmap(_worker_centroid_stratified_sampling_from_file, tasks)

    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])
    return [smiles for _, smiles in results]


def sample_from_cluster_files(cluster_dir,
                               sampling_method: str = 'medoids',
                               centroids: np.ndarray = None,
                               fp_type: str = 'ECFP4',
                               n_bits: int = 2048,
                               sample: bool = True,
                               sample_min_size: int = 1_000,
                               n_processes: int = None):
    """Sample molecules from cluster files using various strategies.

    Parameters
    ----------
    cluster_dir: str or Path
        Directory containing cluster files (.smi or .smi.gz files).
        Each file represents one cluster.
    sampling_method: str, default='medoids'
        Sampling strategy: 'medoids' (most similar compound to cluster center)
        or 'centroid-like' (most similar to precomputed centroid).
    centroids: np.ndarray, optional
        Centroid vectors for 'centroid-like' sampling.
        Shape: (n_clusters, n_bits). Required for 'centroid-like' sampling.
    fp_type: str, default='ECFP4'
        Fingerprint type for computing fingerprints from SMILES.
    n_bits: int, default=2048
        Number of bits in fingerprint vectors.
    sample: bool, default=True
        If True, subsample large clusters before computing medoid/centroid similarity.
    sample_min_size: int, default=1000
        Subsample threshold - clusters larger than this are randomly subsampled.
    n_processes: int, optional
        Number of processes for parallel sampling. Default: min(8, cpu_count()).
        Use n_processes=1 to force sequential mode.

    Returns
    -------
    list[str]
        Sampled SMILES strings from each cluster.
    """
    cluster_dir = Path(cluster_dir)
    if not cluster_dir.is_dir():
        raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")

    # Collect all cluster files
    cluster_files = sorted(cluster_dir.glob("*.smi")) + sorted(cluster_dir.glob("*.smi.gz"))
    if not cluster_files:
        raise FileNotFoundError(f"No .smi or .smi.gz files found in {cluster_dir}")

    # Perform sampling
    if sampling_method == 'medoids':
        return _medoids_sampling_from_files(cluster_files, fp_type, n_bits, sample, sample_min_size, n_processes)
    elif sampling_method == 'centroid-like':
        if centroids is None:
            raise ValueError("Centroids must be provided for centroid-like sampling.")
        if len(centroids) != len(cluster_files):
            raise ValueError(f"Number of centroids ({len(centroids)}) must match number of cluster files ({len(cluster_files)}).")
        return _centroid_like_sampling_from_files(cluster_files, centroids, fp_type, n_bits, sample, sample_min_size, n_processes)
    elif sampling_method == 'centroid-stratified':
        if centroids is None:
            raise ValueError("Centroids must be provided for centroid-stratified sampling.")
        if len(centroids) != len(cluster_files):
            raise ValueError(f"Number of centroids ({len(centroids)}) must match number of cluster files ({len(cluster_files)}).")
        # For centroid-stratified, we can use the same function as centroid-like
        return _centroid_stratified_sampling_from_files(cluster_files, centroids, fp_type, n_bits, sample, sample_min_size, n_processes)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
