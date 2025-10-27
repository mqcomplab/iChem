import numpy as np
from ._input_check import input_check

def calculate_counters(data, n_objects = None) -> dict:
    """Calculate 1-similarity, 0-similarity, and dissimilarity counters

    Arguments
    ---------
    data : np.ndarray
        Array of arrays, each sub-array contains the binary object 
        OR Array with the columnwise sum, if so specify n_objects

    n_objects : int
        Number of objects, only necessary if the column wize sum is the input data.    

    Returns
    -------
    counters : dict
        Dictionary with 1-similarity counters (a), 0-similarity counters (d), total similarity counters (a+d, total_sim),
        total_counters (p), and dissimilarity counters (c+b, total_dis).
        Counters can be used to calculate any similarity index of choice using their respective formulas.

    """
    
    c_total, n_objects = input_check(data, n_objects)

    # Calculate a, d, b + c
    a_array = c_total * (c_total - 1) / 2
    off_coincidences = n_objects - c_total
    d_array = off_coincidences * (off_coincidences - 1) / 2
    dis_array = off_coincidences * c_total

    a = np.sum(a_array)
    d = np.sum(d_array)
    total_dis = np.sum(dis_array)
            
    total_sim = a + d
    p = total_sim + total_dis
    
    counters = {"a": a, "d": d, "total_sim": total_sim,
                "total_dis": total_dis, "p": p}
    
    return counters
