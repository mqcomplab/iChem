import numpy as np # type: ignore
from .isim import calculate_isim

def calculate_medoid(fingerprints, n_ary = 'JT') -> int:
    """Calculate the medoid in a dataset based on complementary similarity.
    Returns the index of the medoid."""
    return np.argmin(calculate_comp_sim(fingerprints, n_ary = n_ary))

def calculate_outlier(fingerprints, n_ary = 'JT') -> int:
    """Calculate the outlier in a dataset based on complementary similarity.
    Returns the index of the outlier."""
    return np.argmax(calculate_comp_sim(fingerprints, n_ary = n_ary))

def comp_sim_indexes(fingerprints: np.ndarray, n_ary: str = 'JT') -> np.ndarray:
    """
    This function computes the complementary similarity for all objects in the dataset.
    
    Parameters
    fingerprints: numpy array of fingerprints
    n_ary: type of similarity to index compute
    
    Returns
    -------
    indexes: numpy array
        array with the rank based on complementary similarity"""
    comp_sim = calculate_comp_sim(fingerprints, n_ary = n_ary)
    indexes = np.argsort(np.argsort(comp_sim))
    
    return indexes

def calculate_comp_sim(fingerprints, n_ary = 'JT') -> np.ndarray:
    """Calculate the complementary similarity for RR, JT, or SM

    Arguments
    ---------
    fingerprints : np.ndarray
        Array of arrays, each sub-array is a binary fingerprint that represents an object/molecule.
        
    n_objects : int
        Number of objects, only necessary if the column wize sum is the input data.

    n_ary : str
        String with the initials of the desired similarity index to calculate the iSIM from. 
        Only RR, JT, or SM are available. For other indexes use gen_sim_dict.

    Returns
    -------
    comp_sims : nd.array
        1D array with the complementary similarities of all the molecules in the set.
    """

    # Define the number of objects, this is one less than the number of total objects because we are excluding the molecule    
    n_objects = len(fingerprints) - 1

    # Get the columnwise sum of the data
    c_total = np.sum(fingerprints, axis = 0)

    comp_sims = [calculate_isim(c_total - fingerprints[i], n_objects = n_objects, n_ary = n_ary) for i in range(len(fingerprints))]
    
    return comp_sims
    
