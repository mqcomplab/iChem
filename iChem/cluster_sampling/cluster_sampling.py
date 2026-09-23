import numpy as np
import pickle as pkl
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from bblean.similarity import jt_isim_medoid, jt_sim_packed, jt_stratified_sampling #type: ignore
from bblean import BitBirch #type: ignore
from ..bitbirch._config import THRESHOLD, BRANCHING_FACTOR, MERGE_CRITERION, N_BITS
from ..utils import binary_fps, load_smiles, load_smiles_gzipped
from ..bitbirch import cluster


def _worker_medoid_sampling(cluster_idx: int, cluster: list[int], fps: np.ndarray,
                             min_size: int, sample: bool, sample_min_size: int) -> tuple[int, int]:
    """Worker function to compute medoid for a single cluster.

    Returns (cluster_idx, sampled_medoid_idx) for order preservation.
    """
    if len(cluster) < min_size:
        return None

    if sample:
        if len(cluster) > sample_min_size:
            random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
        else:
            random_sample = np.array(cluster)
        cluster_fps = fps[random_sample]
        medoid_idx, _ = jt_isim_medoid(cluster_fps)
        return (cluster_idx, random_sample[medoid_idx])
    else:
        cluster_fps = fps[cluster]
        medoid_idx, _ = jt_isim_medoid(cluster_fps)
        return (cluster_idx, cluster[medoid_idx])


def _worker_centroid_like_sampling(cluster_idx: int, cluster: list[int], centroid: np.ndarray,
                                    fps: np.ndarray, sample: bool,
                                    sample_min_size: int) -> tuple[int, int]:
    """Worker function to compute centroid-like for a single cluster.

    Returns (cluster_idx, sampled_idx) for order preservation.
    """
    if sample:
        if len(cluster) > sample_min_size:
            random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
        else:
            random_sample = np.array(cluster)
        cluster_fps = fps[random_sample]
        similarities = jt_sim_packed(centroid, cluster_fps)
        closest_idx = np.argmax(similarities)
        return (cluster_idx, random_sample[closest_idx])
    else:
        cluster_fps = fps[cluster]
        similarities = jt_sim_packed(centroid, cluster_fps)
        closest_idx = np.argmax(similarities)
        return (cluster_idx, cluster[closest_idx])


def _worker_centroid_strat_sampling(cluster_idx: int, cluster: list[int], centroid: np.ndarray,
                                    fps: np.ndarray, sample: bool,
                                    sample_min_size: int) -> list[tuple[int, int, int]]:
    """Worker function to compute centroid-stratified sampling for a single cluster.

    Returns (cluster_idx, sampled_idx, sampled_weight) for order preservation.
    """
    # For tiny clusters, returning all members is faster and avoids redundant similarity work.
    if len(cluster) <= 3:
        return [(cluster_idx, idx, 1) for idx in cluster]

    # Assign each representative an equal coarse weight from its source cluster.
    rep_weight = max(1, int(len(cluster) / 3))

    if sample:
        if len(cluster) > sample_min_size:
            random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
        else:
            random_sample = np.array(cluster)
        cluster_fps = fps[random_sample]
        similarities = jt_sim_packed(centroid, cluster_fps)
        # include the closest to the centroid
        closest_idx = np.argmax(similarities)
        # include the most dissimilar to the centroid
        farthest_idx = np.argmin(similarities)
        # include the mean similarity to the centroid
        mean_idx = np.argmin(np.abs(similarities - np.mean(similarities)))
        return [(cluster_idx, random_sample[closest_idx], rep_weight),
            (cluster_idx, random_sample[farthest_idx], rep_weight),
            (cluster_idx, random_sample[mean_idx], rep_weight)]
    else:
        cluster_fps = fps[cluster]
        similarities = jt_sim_packed(centroid, cluster_fps)
        # include the closest to the centroid
        closest_idx = np.argmax(similarities)
        # include the most dissimilar to the centroid
        farthest_idx = np.argmin(similarities)
        # include the mean similarity to the centroid
        mean_idx = np.argmin(np.abs(similarities - np.mean(similarities)))
        return [(cluster_idx, cluster[closest_idx], rep_weight),
            (cluster_idx, cluster[farthest_idx], rep_weight),
            (cluster_idx, cluster[mean_idx], rep_weight)]
    

def _worker_stratified_sampling(cluster_idx: int, cluster: list[int], fps: np.ndarray,
                             min_size: int, sample: bool, sample_min_size: int) -> list[tuple[int, int, int]]:
    """Worker function to compute stratified sampling for a single cluster.
    """
    cluster_size = len(cluster)
    if cluster_size < min_size:
        return None

    if cluster_size <= 25:
        cluster_fps = fps[cluster]
        medoid_idx, _ = jt_isim_medoid(cluster_fps)
        return [(cluster_idx, cluster[medoid_idx], cluster_size)]
    else:
        if sample:
            # Sample large clusters to reduce computation time
            if cluster_size > sample_min_size:
                random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
            else:
                random_sample = np.array(cluster)
            cluster_fps = fps[random_sample]
        else:
            random_sample = np.array(cluster)
            cluster_fps = fps[random_sample]

        strat_idxs = jt_stratified_sampling(cluster_fps, n_samples=10)
        strat_weights = _stratified_split_weights(cluster_size, 10)
        return [(cluster_idx, random_sample[idx], strat_weights[pos]) for pos, idx in enumerate(strat_idxs)]


def _stratified_split_weights(cluster_size: int, representative_count: int) -> list[int]:
    """Split a cluster size into representative weights that sum exactly to the cluster size."""
    if representative_count <= 0:
        return []
    base_weight, remainder = divmod(cluster_size, representative_count)
    return [base_weight + (1 if idx < remainder else 0) for idx in range(representative_count)]


def _singletons_sampling(clusters: list[list[int]],
                         fps: np.ndarray = None,
                         smiles: list = None):
    """Sample all the singletons in a clustering."""
    singleton_clusters = [cluster for cluster in clusters if len(cluster) == 1]
    singletons_list = [cluster[0] for cluster in singleton_clusters]
    if fps is not None and smiles is not None:
        return fps[singletons_list], [smiles[idx] for idx in singletons_list]
    elif fps is not None:
        return fps[singletons_list]
    elif smiles is not None:
        return [smiles[idx] for idx in singletons_list]
    else:
        return singletons_list
    
    
def _medoids_sampling(clusters: list[list[int]],
                      fps: np.ndarray = None,
                      smiles: list = None,
                      min_size: int = 0,
                      fp_type: str = 'ECFP4',
                      n_bits: int = 2048,
                      sample: bool = True,
                      sample_min_size: int = 1_000,
                      n_processes: int = None):
    """Sample the medoid of each cluster.

    When fps are provided, can use multiprocessing for parallel computation.
    When fps is None, uses sequential mode to avoid nested multiprocessing with binary_fps.
    """
    if fps is not None:
        return _medoids_sampling_parallel(clusters, fps, smiles, min_size, sample, sample_min_size, n_processes)
    else:
        return _medoids_sampling_sequential(clusters, fps, smiles, min_size, fp_type, n_bits, sample, sample_min_size)


def _medoids_sampling_parallel(clusters: list[list[int]], fps: np.ndarray,
                               smiles: list = None, min_size: int = 0,
                               sample: bool = True, sample_min_size: int = 1_000,
                               n_processes: int = None) -> list:
    """Parallel medoid sampling when fps are provided."""
    if n_processes is None or n_processes <= 0:
        n_processes = min(8, cpu_count())

    # Fallback to sequential if only one process requested
    if n_processes == 1:
        return _medoids_sampling_sequential(clusters, fps, smiles, min_size, 'ECFP4', 2048, sample, sample_min_size)

    # Optimize process count based on number of clusters
    n_processes = min(n_processes, len(clusters))

    # Create tasks: (cluster_idx, cluster, fps, min_size, sample, sample_min_size)
    tasks = [(idx, cluster, fps, min_size, sample, sample_min_size)
             for idx, cluster in enumerate(clusters)]

    # Use multiprocessing Pool with progress bar
    show_progress = len(clusters) > 100
    with Pool(processes=n_processes) as pool:
        if show_progress:
            results = list(tqdm(pool.starmap(_worker_medoid_sampling, tasks),
                               total=len(tasks), desc="Sampling medoids"))
        else:
            results = pool.starmap(_worker_medoid_sampling, tasks)

    # Filter out None results (clusters below min_size) and sort by cluster_idx
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])
    medoids_list = [idx for _, idx in results]

    if smiles is not None:
        return [smiles[idx] for idx in medoids_list]
    else:
        return medoids_list


def _medoids_sampling_sequential(clusters: list[list[int]], fps: np.ndarray,
                                 smiles: list = None, min_size: int = 0,
                                 fp_type: str = 'ECFP4', n_bits: int = 2048,
                                 sample: bool = True, sample_min_size: int = 1_000) -> list:
    """Sequential medoid sampling (original implementation)."""
    if fps is not None:
        medoids_list = []
        for cluster in clusters:
            if len(cluster) >= min_size:
                if sample:
                    if len(cluster) > sample_min_size:
                        random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                    else:
                        random_sample = np.array(cluster)
                    cluster_fps = fps[random_sample]
                    medoid_idx, _ = jt_isim_medoid(cluster_fps)
                    medoids_list.append(random_sample[medoid_idx])
                else:
                    cluster_fps = fps[cluster]
                    medoid_idx, _ = jt_isim_medoid(cluster_fps)
                    medoids_list.append(cluster[medoid_idx])
    else:
        medoids_list = []
        for cluster in clusters:
            if len(cluster) >= min_size:
                if sample:
                    if len(cluster) > sample_min_size:
                        random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                    else:
                        random_sample = np.array(cluster)
                    cluster_smiles = [smiles[idx] for idx in random_sample] if smiles else None
                    cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                    medoid_idx, _ = jt_isim_medoid(cluster_fps)
                    medoids_list.append(random_sample[medoid_idx])
                else:
                    cluster_smiles = [smiles[idx] for idx in cluster] if smiles else None
                    cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                    medoid_idx, _ = jt_isim_medoid(cluster_fps)
                    medoids_list.append(cluster[medoid_idx])
    if smiles is not None:
        return [smiles[idx] for idx in medoids_list]
    else:
        return medoids_list


def _centroid_like_sampling(clusters: list[list[int]],
                           centroids: np.ndarray,
                           fps: np.ndarray = None,
                           smiles: list = None,
                           fp_type: str = 'ECFP4',
                           n_bits: int = 2048,
                           sample: bool = True,
                           sample_min_size: int = 1_000,
                           n_processes: int = None):
    """Sample the centroid-like of each cluster.

    When fps are provided, can use multiprocessing for parallel computation.
    When fps is None, uses sequential mode to avoid nested multiprocessing with binary_fps.
    """
    if centroids is None:
        raise ValueError("Centroids are required for centroid-like sampling.")

    if fps is not None:
        return _centroid_like_sampling_parallel(clusters, centroids, fps, smiles, sample, sample_min_size, n_processes)
    else:
        return _centroid_like_sampling_sequential(clusters, centroids, fps, smiles, fp_type, n_bits, sample, sample_min_size)


def _centroid_like_sampling_parallel(clusters: list[list[int]], centroids: np.ndarray,
                                     fps: np.ndarray, smiles: list = None,
                                     sample: bool = True, sample_min_size: int = 1_000,
                                     n_processes: int = None) -> list:
    """Parallel centroid-like sampling when fps are provided."""
    if n_processes is None or n_processes <= 0:
        n_processes = min(8, cpu_count())

    # Fallback to sequential if only one process requested
    if n_processes == 1:
        return _centroid_like_sampling_sequential(clusters, centroids, fps, smiles, 'ECFP4', 2048, sample, sample_min_size)

    # Optimize process count based on number of clusters
    n_processes = min(n_processes, len(clusters))

    # Create tasks: (cluster_idx, cluster, centroid, fps, sample, sample_min_size)
    tasks = [(idx, cluster, centroids[idx], fps, sample, sample_min_size)
             for idx, cluster in enumerate(clusters)]

    # Use multiprocessing Pool with progress bar
    show_progress = len(clusters) > 100
    with Pool(processes=n_processes) as pool:
        if show_progress:
            results = list(tqdm(pool.starmap(_worker_centroid_like_sampling, tasks),
                               total=len(tasks), desc="Sampling centroid-like"))
        else:
            results = pool.starmap(_worker_centroid_like_sampling, tasks)

    # Sort by cluster_idx to preserve order
    results.sort(key=lambda x: x[0])
    centroid_like_list = [idx for _, idx in results]

    if smiles is not None:
        return [smiles[idx] for idx in centroid_like_list]
    else:
        return centroid_like_list


def _centroid_like_sampling_sequential(clusters: list[list[int]], centroids: np.ndarray,
                                       fps: np.ndarray = None, smiles: list = None,
                                       fp_type: str = 'ECFP4', n_bits: int = 2048,
                                       sample: bool = True, sample_min_size: int = 1_000) -> list:
    """Sequential centroid-like sampling (original implementation)."""
    centroid_like_list = []
    for cluster, centroid in zip(clusters, centroids):
        if fps is not None:
            if sample:
                if len(cluster) > sample_min_size:
                    random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                else:
                    random_sample = np.array(cluster)
                cluster_fps = fps[random_sample]
                similarities = jt_sim_packed(centroid, cluster_fps)
                closest_idx = np.argmax(similarities)
                centroid_like_list.append(random_sample[closest_idx])
            else:
                cluster_fps = fps[cluster]
                similarities = jt_sim_packed(centroid, cluster_fps)
                closest_idx = np.argmax(similarities)
                centroid_like_list.append(cluster[closest_idx])
        else:
            if sample:
                if len(cluster) > sample_min_size:
                    random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                else:
                    random_sample = np.array(cluster)
                cluster_smiles = [smiles[idx] for idx in random_sample] if smiles else None
                cluster_fps = binary_fps(
                    cluster_smiles,
                    fp_type=fp_type,
                    n_bits=n_bits,
                    return_invalid=False,
                    packed=True,
                )
                similarities = jt_sim_packed(cluster_fps, centroid)
                closest_idx = np.argmax(similarities)
                centroid_like_list.append(random_sample[closest_idx])
            else:
                cluster_smiles = [smiles[idx] for idx in cluster] if smiles else None
                cluster_fps = binary_fps(
                    cluster_smiles,
                    fp_type=fp_type,
                    n_bits=n_bits,
                    return_invalid=False,
                    packed=True,
                )
                similarities = jt_sim_packed(cluster_fps, centroid)
                closest_idx = np.argmax(similarities)
                centroid_like_list.append(cluster[closest_idx])
    if smiles:
        return [smiles[idx] for idx in centroid_like_list]
    else:
        return centroid_like_list

def centroid_stratified_sampling(clusters: list[list[int]],
                                 centroids: np.ndarray,
                                 fps: np.ndarray = None,
                                 smiles: list = None,
                                 fp_type: str = 'ECFP4',
                                 n_bits: int = 2048,
                                 sample: bool = True,
                                 sample_min_size: int = 1_000,
                                 n_processes: int = None,
                                 return_weights: bool = True):
    """Sample the closest, farthest, and mean similarity to centroid for each cluster.

    When fps are provided, can use multiprocessing for parallel computation.
    When fps is None, uses sequential mode to avoid nested multiprocessing with binary_fps.
    """
    if centroids is None:
        raise ValueError("Centroids are required for centroid-stratified sampling.")

    if fps is not None:
        return _centroid_stratified_sampling_parallel(clusters, centroids, fps, smiles, sample, sample_min_size, n_processes, return_weights)
    else:
        return _centroid_stratified_sampling_sequential(clusters, centroids, fps, smiles, fp_type, n_bits, sample, sample_min_size, return_weights)
    

def _centroid_stratified_sampling_parallel(clusters: list[list[int]], centroids: np.ndarray,
                                             fps: np.ndarray, smiles: list = None,
                                             sample: bool = True, sample_min_size: int = 1_000,
                                             n_processes: int = None,
                                             return_weights: bool = True):
    """Parallel centroid-stratified sampling when fps are provided."""
    if n_processes is None or n_processes <= 0:
        n_processes = min(8, cpu_count())

    # Fallback to sequential if only one process requested
    if n_processes == 1:
        return _centroid_stratified_sampling_sequential(
            clusters,
            centroids,
            fps,
            smiles,
            'ECFP4',
            2048,
            sample,
            sample_min_size,
            return_weights,
        )

    # Optimize process count based on number of clusters
    n_processes = min(n_processes, len(clusters))

    # Create tasks: (cluster_idx, cluster, centroid, fps, sample, sample_min_size)
    tasks = [(idx, cluster, centroids[idx], fps, sample, sample_min_size)
             for idx, cluster in enumerate(clusters)]

    # Use multiprocessing Pool with progress bar
    show_progress = len(clusters) > 100
    with Pool(processes=n_processes) as pool:
        if show_progress:
            results = list(tqdm(pool.starmap(_worker_centroid_strat_sampling, tasks),
                                total=len(tasks), desc="Sampling centroid-stratified"))
        else:
            results = pool.starmap(_worker_centroid_strat_sampling, tasks)

    # Flatten the list of lists and sort by cluster_idx to preserve order
    flattened_results = [item for sublist in results for item in sublist]
    flattened_results.sort(key=lambda x: x[0])
    stratified_list = [idx for _, idx, _ in flattened_results]
    stratified_weights = [weight for _, _, weight in flattened_results]

    if smiles is not None:
        stratified_list = [smiles[idx] for idx in stratified_list]

    if return_weights:
        return stratified_list, stratified_weights
    return stratified_list


def _centroid_stratified_sampling_sequential(clusters: list[list[int]], centroids: np.ndarray,
                                                fps: np.ndarray = None, smiles: list = None,
                                                fp_type: str = 'ECFP4', n_bits: int = 2048,
                                                sample: bool = True, sample_min_size: int = 1_000,
                                                return_weights: bool = True):
        """Sequential centroid-stratified sampling (original implementation)."""
        stratified_list = []
        stratified_weights = []
        for cluster, centroid in zip(clusters, centroids):
            # For tiny clusters, keep all members and skip similarity calculations.
            if len(cluster) <= 3:
                stratified_list.extend(cluster)
                stratified_weights.extend([1] * len(cluster))
                continue

            rep_weight = max(1, int(len(cluster) / 3))

            if fps is not None:
                if sample:
                        if len(cluster) > sample_min_size:
                            random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                        else:
                            random_sample = np.array(cluster)
                        cluster_fps = fps[random_sample]
                        similarities = jt_sim_packed(centroid, cluster_fps)
                else:
                        cluster_fps = fps[cluster]
                        similarities = jt_sim_packed(centroid, cluster_fps)
            else:
                if sample:
                        if len(cluster) > sample_min_size:
                            random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                        else:
                            random_sample = np.array(cluster)
                        cluster_smiles = [smiles[idx] for idx in random_sample] if smiles else None
                        cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                        similarities = jt_sim_packed(centroid, cluster_fps)
                else:
                        cluster_smiles = [smiles[idx] for idx in cluster] if smiles else None
                        cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                        similarities = jt_sim_packed(centroid, cluster_fps)
    
            # Include the closest to the centroid
            closest_idx = np.argmax(similarities)
            # Include the most dissimilar to the centroid
            farthest_idx = np.argmin(similarities)
            # Include the mean similarity to the centroid
            mean_idx = np.argmin(np.abs(similarities - np.mean(similarities)))
    
            if sample:
                stratified_list.append(random_sample[closest_idx])
                stratified_list.append(random_sample[farthest_idx])
                stratified_list.append(random_sample[mean_idx])
            else:
                stratified_list.append(cluster[closest_idx])
                stratified_list.append(cluster[farthest_idx])
                stratified_list.append(cluster[mean_idx])
            stratified_weights.extend([rep_weight, rep_weight, rep_weight])
    
        if smiles is not None:
            stratified_list = [smiles[idx] for idx in stratified_list]

        if return_weights:
            return stratified_list, stratified_weights
        return stratified_list


def _stratified_sampling(clusters: list[list[int]],
                         fps: np.ndarray = None,
                         smiles: list = None,
                         min_size: int = 0,
                         fp_type: str = 'ECFP4',
                         n_bits: int = 2048,
                         sample: bool = True,
                         sample_min_size: int = 1_000,
                         n_processes: int = None,
                         return_weights: bool = True):
    """Sample 3 representatives from each cluster using stratified sampling.
    """
    if fps is not None:
        return _stratified_sampling_parallel(clusters, fps, smiles, min_size, sample, sample_min_size, n_processes, return_weights)
    else:
        return _stratified_sampling_sequential(clusters, fps, smiles, min_size, fp_type, n_bits, sample, sample_min_size, return_weights)
    

def _stratified_sampling_sequential(clusters: list[list[int]], fps: np.ndarray,
                                    smiles: list = None, min_size: int = 0,
                                    fp_type: str = 'ECFP4', n_bits: int = 2048,
                                    sample: bool = True, sample_min_size: int = 1_000,
                                    return_weights: bool = True):
    """Sequential stratified sampling (original implementation)."""
    stratified_list = []
    stratified_weights = []
    for cluster in clusters:
        cluster_size = len(cluster)
        if cluster_size < min_size:
            continue

        # For small clusters, keep one member and skip similarity calculations.
        if cluster_size <= 5:
            if fps is not None:
                cluster_fps = fps[cluster]
            else:
                cluster_smiles = [smiles[idx] for idx in cluster] if smiles else None
                cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
            medoid_idx, _ = jt_isim_medoid(cluster_fps)
            stratified_list.append(cluster[medoid_idx])
            stratified_weights.append(cluster_size)
            continue

        strat_weights = _stratified_split_weights(cluster_size, 3)

        if fps is not None:
            if sample:
                if cluster_size > sample_min_size:
                    random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                else:
                    random_sample = np.array(cluster)
                cluster_fps = fps[random_sample]
                strat_idxs = jt_stratified_sampling(cluster_fps, n_samples=3)
                stratified_list.extend(random_sample[strat_idxs])
            else:
                cluster_fps = fps[cluster]
                strat_idxs = jt_stratified_sampling(cluster_fps, n_samples=3)
                stratified_list.extend(np.array(cluster)[strat_idxs])
        else:
            if sample:
                if cluster_size > sample_min_size:
                    random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                else:
                    random_sample = np.array(cluster)
                cluster_smiles = [smiles[idx] for idx in random_sample] if smiles else None
                cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                strat_idxs = jt_stratified_sampling(cluster_fps, n_samples=3)
                stratified_list.extend(random_sample[strat_idxs])
            else:
                cluster_smiles = [smiles[idx] for idx in cluster] if smiles else None
                cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                strat_idxs = jt_stratified_sampling(cluster_fps, n_samples=3)
                stratified_list.extend(np.array(cluster)[strat_idxs])

            stratified_weights.extend(strat_weights)

    if smiles is not None:
        stratified_list = [smiles[idx] for idx in stratified_list]
    
    if return_weights:
        return stratified_list, stratified_weights
    return stratified_list



def _stratified_sampling_parallel(clusters: list[list[int]], fps: np.ndarray,
                                  smiles: list = None, min_size: int = 0,
                                  sample: bool = True, sample_min_size: int = 1_000,
                                  n_processes: int = None, return_weights: bool = True) -> list:
    """Parallel stratified sampling when fps are provided."""
    if n_processes is None or n_processes <= 0:
        n_processes = min(8, cpu_count())

    # Fallback to sequential if only one process requested
    if n_processes == 1:
        return _stratified_sampling_sequential(clusters, fps, smiles, min_size, 'ECFP4', 2048, sample, sample_min_size, return_weights)
    
    # Optimize process count based on number of clusters
    n_processes = min(n_processes, len(clusters))

    # Create tasks: (cluster_idx, cluster, fps, min_size, sample, sample_min_size)
    tasks = [(idx, cluster, fps, min_size, sample, sample_min_size)
             for idx, cluster in enumerate(clusters)]

    # Use multiprocessing Pool with progress bar
    show_progress = len(clusters) > 100
    with Pool(processes=n_processes) as pool:
        if show_progress:
            results = list(tqdm(pool.starmap(_worker_stratified_sampling, tasks),
                                total=len(tasks), desc="Sampling centroid-stratified"))
        else:
            results = pool.starmap(_worker_stratified_sampling, tasks)

    # Flatten the list of lists and sort by cluster_idx to preserve order
    flattened_results = [item for sublist in results for item in sublist]
    flattened_results.sort(key=lambda x: x[0])
    stratified_list = [idx for _, idx, _ in flattened_results]
    stratified_weights = [weight for _, _, weight in flattened_results]

    if smiles is not None:
        stratified_list = [smiles[idx] for idx in stratified_list]

    if return_weights:
        return stratified_list, stratified_weights
    return stratified_list



def sample_clusters(clusters,
                     sampling_method: str = 'medoids',
                     fps = None,
                     smiles = None,
                     centroids = None,
                     min_size: int = 0,
                     fp_type: str = 'ECFP4',
                     n_bits: int = 2048,
                     sample: bool = True,
                     sample_min_size: int = 1000,
                     n_processes: int = None,
                     return_weights: bool = True):
    """Sample molecules from precomputed clusters using various strategies.

    Parameters
    ----------
    clusters: str, Path, or list of lists
        Cluster assignments. Can be:
        - Path to pickled file containing clusters (list of lists)
        - Pre-loaded list of lists with cluster assignments
    sampling_method: str, default='medoids'
        Sampling strategy: 'singletons' (all single-element clusters),
        'medoids' (most similar compound to cluster median),
        or 'centroid-like' (most similar to precomputed centroid).
    fps: str, Path, or np.ndarray, optional
        Fingerprints data. Can be:
        - Path to .npy file or directory of .npy files
        - Pre-loaded numpy array (shape: n_molecules x n_bits)
    smiles: str, Path, or list, optional
        SMILES data. Can be:
        - Path to .smi or .smi.gz file or directory containing them
        - Pre-loaded list of SMILES strings
    centroids: str, Path, or np.ndarray, optional
        Centroid vectors. Can be:
        - Path to pickled file containing centroids
        - Pre-loaded numpy array (shape: n_clusters x n_bits)
        Required for 'centroid-like' sampling.
    min_size: int, default=0
        Minimum cluster size to sample from (for 'medoids' method).
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
        Only applies when fps are provided (to avoid nested multiprocessing with binary_fps).
        Use n_processes=1 to force sequential mode.

    Returns
    -------
    list
        Sampled molecule indices or SMILES strings, depending on input data provided.
    """
    # Load clusters: either path or direct list
    if isinstance(clusters, list):
        pass  # Already a list
    elif isinstance(clusters, (str, Path)):
        clusters_path = Path(clusters)
        if clusters_path.is_file():
            clusters = pkl.load(open(clusters_path, 'rb'))
        else:
            raise FileNotFoundError(
                f"Clusters file not found: {clusters_path}"
            )

    # Handle fingerprints: either path or direct np.ndarray
    if isinstance(fps, np.ndarray):
        pass  # Already a numpy array
    elif fps is not None:
        fps_path = Path(fps)
        if fps_path.is_file():
            fps = np.load(fps_path, mmap_mode='r')
        elif fps_path.is_dir():
            fps_files = sorted(fps_path.glob("*.npy"))
            fps_list = [np.load(fp_file, mmap_mode='r') for fp_file in fps_files]
            fps = np.concatenate(fps_list, axis=0)
        else:
            fps = None
    else:
        fps = None

    # Handle smiles: either path or direct list
    if isinstance(smiles, list):
        pass  # Already a list
    elif smiles is not None:
        smiles_path = Path(smiles)
        if smiles_path.is_file():
            if smiles_path.suffix == '.smi':
                smiles = load_smiles(smiles_path)
            elif smiles_path.suffix == '.smi.gz':
                smiles = load_smiles_gzipped(smiles_path)
            else:
                smiles = None
        elif smiles_path.is_dir():
            smiles_files_smi = sorted(smiles_path.glob("*.smi"))
            smiles_files_gz = sorted(smiles_path.glob("*.smi.gz"))
            smiles = []
            for smi_file in smiles_files_smi:
                smiles.extend(load_smiles(smi_file))
            for smi_file in smiles_files_gz:
                smiles.extend(load_smiles_gzipped(smi_file))
        else:
            smiles = None
    else:
        smiles = None

    # Handle centroids: either path or direct np.ndarray
    if isinstance(centroids, np.ndarray):
        pass  # Already a numpy array
    elif centroids is not None:
        centroids_path = Path(centroids)
        if centroids_path.is_file():
            centroids = pkl.load(open(centroids_path, 'rb'))
        else:
            centroids = None
    else:
        centroids = None

    # Perform sampling
    if sampling_method == 'singletons':
        return _singletons_sampling(clusters,
                                    fps,
                                    smiles)
    elif sampling_method == 'medoids':
        return _medoids_sampling(clusters=clusters,
                                 fps=fps,
                                 smiles=smiles,
                                 min_size=min_size,
                                 fp_type=fp_type,
                                 n_bits=n_bits,
                                 sample=sample,
                                 sample_min_size=sample_min_size,
                                 n_processes=n_processes)
    elif sampling_method == 'centroid-like':
        if centroids is None:
            raise ValueError(
                "Centroids must be provided for centroid-like sampling."
            )
        return _centroid_like_sampling(clusters=clusters,
                                       centroids=centroids,
                                       fps=fps,
                                       smiles=smiles,
                                       fp_type=fp_type,
                                       n_bits=n_bits,
                                       sample=sample,
                                       sample_min_size=sample_min_size,
                                       n_processes=n_processes)
    elif sampling_method == 'centroid-stratified':
        if centroids is None:
            raise ValueError(
                "Centroids must be provided for centroid-stratified sampling."
            )
        return_weights = True
        return centroid_stratified_sampling(clusters=clusters,
                                            centroids=centroids,
                                            fps=fps,
                                            smiles=smiles,
                                            fp_type=fp_type,
                                            n_bits=n_bits,
                                            sample=sample,
                                            sample_min_size=sample_min_size,
                                            n_processes=n_processes,
                                            return_weights=return_weights)
    elif sampling_method == 'stratified':
        return_weights = True
        return _stratified_sampling(clusters=clusters,
                                    fps=fps,
                                    smiles=smiles,
                                    min_size=min_size,
                                    fp_type=fp_type,
                                    n_bits=n_bits,
                                    sample=sample,
                                    sample_min_size=sample_min_size,
                                    n_processes=n_processes,
                                    return_weights=return_weights)
    else:
        raise ValueError(
            f"Invalid sampling method: {sampling_method}. "
            "Choose from 'singletons', 'medoids', 'centroid-like', or 'centroid-stratified'."
        )
    
def bitbirch_sampling(fps: np.ndarray = None,
                      smiles: list = None,
                      sampling_method: str = 'medoids',
                      min_size: int = 0,
                      fp_type: str = 'ECFP4',
                      n_bits: int = N_BITS,
                      threshold: float = THRESHOLD,
                      branching_factor: int = BRANCHING_FACTOR,
                      merge_criterion: str = MERGE_CRITERION,
                      sample: bool = True,
                      sample_min_size: int = 1_000,
                      return_sizes: bool = True) -> list:
    """Sample using BitBirch clustering medoids directly from fingeprrints and/or smiles.
    This function does the sequential clustering too.
    
    Parameters
    ----------
    fps: np.ndarray, optional
        Precomputed fingerprint array (shape: n_molecules x n_bits).
    smiles: list, optional
        List of SMILES strings corresponding to the fingerprints.
    sampling_method: str, default='medoids'
        Sampling strategy: 'singletons' (all single-element clusters),
        'medoids' (most similar compound to cluster median),
        or 'centroid-like' (most similar to precomputed centroid).
    min_size: int, default=0
        Minimum cluster size to sample from (for 'medoids' method).
    fp_type: str, default='ECFP4'
        Fingerprint type for computing fingerprints from SMILES if fps not provided.
    n_bits: int, default=2048
        Number of bits in fingerprint vectors if fps not provided.
    threshold: float, default=THRESHOLD
        Distance threshold for BitBirch clustering.
    branching_factor: int, default=BRANCHING_FACTOR
        Branching factor for BitBirch clustering.
    merge_criterion: str, default=MERGE_CRITERION
        Merge criterion for BitBirch clustering.
    sample: bool, default=True
        If True, subsample large clusters before computing medoid/centroid similarity.
    sample_min_size: int, default=1000
        Subsample threshold - clusters larger than this are randomly subsampled.
        
    Returns
    -------
    list
        Sampled molecule indices or SMILES strings, depending on input data provided."""

    # Generate fingerprints if not provided, but smiles are available
    if fps is None and smiles is None:
        raise ValueError("At least one of fps or smiles must be provided for BitBirch sampling.")
    if fps is None:
        fps, invalid = binary_fps(smiles, fp_type=fp_type, n_bits=n_bits)
        if invalid:
            raise ValueError(f"Invalid SMILES found at indices: {invalid}. Please check your SMILES data.")

    # Save the temporatrly the fps
    temp_fps_path = Path("temp_fps.npy")
    np.save(temp_fps_path, fps)

    # Keep size metadata always enabled for downstream libcomp compatibility.
    return_sizes = True

    clusters = cluster(temp_fps_path,
                       threshold=threshold,
                       branching_factor=branching_factor,
                       merge_criterion=merge_criterion,
                       save_centroids=True if sampling_method == 'centroid-like' or sampling_method == 'centroid-stratified' else False)

    if return_sizes and sampling_method in {'centroid-stratified', 'stratified'}:
        samples, stratified_weights = sample_clusters(clusters=clusters,
                                                      sampling_method=sampling_method,
                                                      fps=fps,
                                                      smiles=smiles,
                                                      centroids=Path("cluster-centroids-packed.pkl") if sampling_method == 'centroid-like' or sampling_method == 'centroid-stratified' else None,
                                                      min_size=min_size,
                                                      fp_type=fp_type,
                                                      n_bits=n_bits,
                                                      sample=sample,
                                                      sample_min_size=sample_min_size,
                                                      return_weights=True)
    else:
        samples = sample_clusters(clusters=clusters,
                                  sampling_method=sampling_method,
                                  fps=fps,
                                  smiles=smiles,
                                  centroids=Path("cluster-centroids-packed.pkl") if sampling_method == 'centroid-like' or sampling_method == 'centroid-stratified' else None,
                                  min_size=min_size,
                                  fp_type=fp_type,
                                  n_bits=n_bits,
                                  sample=sample,
                                  sample_min_size=sample_min_size)
    
    # Delete the temporary fps file
    if temp_fps_path.exists():
        temp_fps_path.unlink()
    if Path("cluster-centroids-packed.pkl").exists():
        Path("cluster-centroids-packed.pkl").unlink()
    if Path("clusters.pkl").exists():
        Path("clusters.pkl").unlink()
    
    if return_sizes:
        if sampling_method == 'singletons':
            cluster_sizes = [1] * len(samples)
        elif sampling_method not in {'centroid-stratified', 'stratified'}:
            cluster_sizes = [len(cluster) for cluster in clusters if len(cluster) >= min_size]
        elif sampling_method in {'centroid-stratified', 'stratified'}:
            cluster_sizes = stratified_weights

        return samples, cluster_sizes
    return samples
