import numpy as np # type: ignore
import pandas as pd # type: ignore
from joblib import Parallel, delayed, parallel_backend # type: ignore
from .sampling import stratified_sampling
from ..utils import pairwise_average, rdkit_pairwise_sim
from .real import calculate_comp_sim_real, pairwise_average_real

def get_stdev_russell_fast(arr):
    """
    Method to obtain the standard deviation of RR similarities of a set of fingerprints in O(NM^2) time complexity.

    Parameters
    ----------
    arr : np.array

    Returns
    -------
    float
    """
    sums = np.sum(arr, axis=0)
    total = len(arr)*(len(arr)-1)/2
    probs = sums*(sums-1)/2/total

    #Covariance Step

    def get_covariance(i):
        output = []
        for j in range(i+1, len(arr[0])):
            counter = 0
            counter += arr[:, i] @ arr[:, j]
            prob = counter*(counter - 1)/2/total
            output.append(prob - probs[i]*probs[j])
        return np.sum(output)
    
    with parallel_backend('loky', n_jobs=10):
        covariances = Parallel()(delayed(get_covariance)(i) for i in range(len(arr[0])))
    
    covariance_sum = np.sum(covariances)

    return np.sqrt(np.sum(probs*(1-probs)) + 2*covariance_sum)/len(arr[0])

def get_stdev_tanimoto_fast(arr):
    """"
    Method to obtain the standard deviation of Tanimoto similarities of a set of fingerprints in O(NM^2) time complexity.
    Due to the nature of the Tanimoto similarity, the results are not as accurate as the Russell-Rao similarity
    
    Parameters
    ----------
    arr : np.array
    
    Returns
    -------
    float
    """
    sums = np.sum(arr, axis=0)
    total = len(arr)*(len(arr)-1)/2
    probs = sums*(sums-1)/2/total

    #Covariance Step

    def get_covariance(i):
        output = []
        for j in range(i+1, len(arr[0])):
            counter = 0
            counter += arr[:, i] @ arr[:, j]
            prob = counter*(counter - 1)/2/total
            output.append(prob - probs[i]*probs[j])
        return np.sum(output)
    
    with parallel_backend('loky', n_jobs=10):
        covariances = Parallel()(delayed(get_covariance)(i) for i in range(len(arr[0])))
    
    covariance_sum = np.sum(covariances)

    ### Getting Denominator
    #Crude approximation
    sums_zeros = len(arr) - sums
    denom = np.sum(total - sums_zeros*(sums_zeros-1)/2)/total
    return np.sqrt(np.sum(probs*(1-probs)) + 2*covariance_sum)/denom

def get_stdev_sokal_fast(arr):
    """"
    Method to obtain the standard deviation of Sokal-Michener similarities of a set of fingerprints in O(NM^2) time complexity."
    
    Parameters
    ----------
    arr : np.array

    Returns
    -------
    float
    """
    sums = np.sum(arr, axis=0)
    total = len(arr)*(len(arr)-1)/2
    probs = sums*(sums-1)/2/total

    sums_zeros = len(arr) - sums
    probs += sums_zeros*(sums_zeros-1)/2/total
    #Covariance Step

    def get_covariance(i):
        output = []
        for j in range(i+1, len(arr[0])):
            counter = 0
            counter += arr[:, i] @ arr[:, j]
            prob = counter*(counter - 1)/2/total
            counter_zeros = 0
            counter_zeros += (1-arr[:, i]) @ (1-arr[:, j])
            prob += counter_zeros*(counter_zeros-1)/2/total

            counter_pair_1 = arr[:, i] @ (1 - arr[:, j])
            counter_pair_2 = (1-arr[:,i]) @ arr[:, j]
            prob += counter_pair_1*(counter_pair_1 - 1)/2/total
            prob += counter_pair_2*(counter_pair_2 - 1)/2/total
            output.append(prob - probs[i]*probs[j])
        return np.sum(output)
    
    with parallel_backend('loky', n_jobs=10):
        covariances = Parallel()(delayed(get_covariance)(i) for i in range(len(arr[0])))
    covariance_sum = np.sum(covariances)
    #print(covariance_sum)

    return np.sqrt(np.sum(probs*(1-probs)) + 2*covariance_sum)/len(arr[0])

def stratified_sigma(fps, n = 50, n_ary = 'JT'):
    """
    Method to estimate the standard deviation by sampling representative fingerprints using stratified sampling.
    Once the sampled is donde the pairwise average is calculated and the standard deviation is estimated.
    
    Parameters
    ----------
    fps : np.array
    
    n : int
        Number of samples to take.
        
    n_ary : str
        Type of similarity to calculate the pairwise average.
        
    Returns
    -------
    standard deviation: float
    """

    # Sample the representative molecules 
    indexes_strat = stratified_sampling(fps, n_ary = n_ary, n_sample = n)
    fps_strat = fps[indexes_strat]

    if n_ary == 'JT':
        # Calculate the pairwise average of the sampled indexes
        average, std = rdkit_pairwise_sim(fps_strat, return_std = True)
    else:
        # Calculate the pairwise average of the sampled indexes
        average, std = pairwise_average(fps_strat, n_ary = n_ary, return_std = True)

    return std

def stratified_sigma_real(fps, n = 50, n_ary = 'JT'):
    """
    Method to estimate the standard deviation by sampling representative fingerprints using stratified sampling for real or count fingerprints.
    
    Parameters
    ----------
    fps : np.array
        Fingerprints should be normalized before being passed to this function.
    
    n : int
        Number of samples to take.
        
    n_ary : str
        Type of similarity to calculate the pairwise average.
        
    Returns
    -------
    standard deviation: float
    """
    # Calculate the complementary similarity for the real fingerprints
    comp_sim_real = calculate_comp_sim_real(fps, n_ary = n_ary)

    indexes_strat = stratified_sampling(comp_sim=comp_sim_real, n_ary = n_ary, n_sample = n)

    fps_strat = fps[indexes_strat]

    _, std = pairwise_average_real(fps_strat, n_ary = n_ary, return_std = True)

    return std

def random_sigma(fps, n = 50, n_ary = 'JT'):
    """"
    Method to estimate the standard deviation by sampling randomly fingerprints.

    Parameters
    ----------
    fps : np.array

    n : int
        Number of samples to take.

    n_ary : str
        Type of similarity to calculate the pairwise average.

    Returns
    -------
    standard deviation: float
    """
    # Sample random molecules
    indexes_rand = np.random.choice(len(fps), n, replace = False)
    fps_rand = fps[indexes_rand]

    if n_ary == 'JT':
        # Calculate the pairwise average of the sampled indexes
        average, std = rdkit_pairwise_sim(fps_rand, return_std = True)
    else:
        # Calculate the pairwise average of the sampled indexes
        average, std = pairwise_average(fps_rand, n_ary = n_ary, return_std = True)

    return std