import numpy as np # type: ignore
from scipy import sparse # type: ignore
from ..iSIM.real import calculate_isim_real
from ..iSIM.sigma import stratified_sigma_real

def calc_centroid(linear_sum, n_samples):
    """Calculates centroid
    
    Parameters
    ----------
    
    linear_sum : np.ndarray
                 Sum of the elements column-wise
    n_samples : int
                Number of samples
                
    Returns
    -------
    centroid : np.ndarray
               Centroid fingerprints of the given set
    """
    return linear_sum/n_samples

def max_separation(X):
    """Finds two objects in X that are very separated
    This is an approximation (not guaranteed to find
    the two absolutely most separated objects), but it is
    a very robust O(N) implementation. Quality of clustering
    does not diminish in the end.
    
    Algorithm:
    a) Find centroid of X
    b) mol1 is the molecule most distant from the centroid
    c) mol2 is the molecule most distant from mol1
    
    Returns
    -------
    (mol1, mol2) : (int, int)
                   indices of mol1 and mol2
    sims_mol1 : np.ndarray
                   Distances to mol1
    sims_mol2: np.ndarray
                   Distances to mol2
    These are needed for node1_dist and node2_dist in _split_node
    """
    # Get the centroid of the set
    n_samples = len(X)
    linear_sum = np.sum(X, axis = 0)
    centroid = calc_centroid(linear_sum, n_samples)

    # Get the similarity of each molecule to the centroid
    mols_dot_products = np.sum(X**2, axis = 1)
    mols_centroid_dot_products = np.dot(X, centroid)
    centroid_dot_product = np.sum(centroid**2)
    sims_med = mols_centroid_dot_products / (mols_dot_products + centroid_dot_product - mols_centroid_dot_products)

    # Get the least similar molecule to the centroid
    mol1 = np.argmin(sims_med)

    # Get the similarity of each molecule to mol1
    mol1_mol1_dot_product = mols_dot_products[mol1]
    mols_mol1_dot_products = np.dot(X, X[mol1])
    sims_mol1 = mols_mol1_dot_products / (mols_dot_products + mol1_mol1_dot_product - mols_mol1_dot_products)

    # Get the least similar molecule to mol1
    mol2 = np.argmin(sims_mol1)

    # Get the similarity of each molecule to mol2
    mol2_mol2_dot_product = mols_dot_products[mol2]
    mols_mol2_dot_products = np.dot(X, X[mol2])
    sims_mol2 = mols_mol2_dot_products / (mols_dot_products + mol2_mol2_dot_product - mols_mol2_dot_products)
    
    return (mol1, mol2), sims_mol1, sims_mol2

def jt_isim_real(linear_sum, sq_sum, n_samples):
    """Calculates the iSIM value for a given set of fingerprints using the linear sum, squared sum, and number of samples.
    
    Parameters
    ----------
    
    linear_sum : np.ndarray
                 Sum of the elements column-wise
    sq_sum : np.ndarray
             Sum of the squares of the elements column-wise
    n_samples : int
                Number of samples

    Returns
    -------
    iSIM : float
           iSIM value for the given set of fingerprints
    """
    ij_array = 0.5 * (linear_sum ** 2 - sq_sum)
    ij = np.sum(ij_array)
    inners = (n_samples - 1) * np.sum(sq_sum)
    
    return ij/(inners - ij)

def optimal_threshold_real(fps, factor = 3.5, n_ary = 'JT'):
    """This function calculates the optimal threshold for the real-valued fingerprints using the iSIM index.
    
    Parameters
    ----------
    
    fps : np.ndarray
          numpy array of fingerprints with real-valued numbers
    factor : float
             factor to multiply the average similarity to get the threshold (default is 3.5)
    n_ary : str
            type of similarity index to calculate [JT, RR, SM] (default is JT)

    Returns
    -------
    threshold : float
                optimal threshold for the given set of fingerprints
    """
    
    isim_real = calculate_isim_real(fps, n_ary = n_ary)
    std_real = stratified_sigma_real(fps, n = 50, n_ary = n_ary)

    threshold = isim_real + factor * std_real

    return threshold


