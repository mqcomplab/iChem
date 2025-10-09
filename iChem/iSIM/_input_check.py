import numpy as np

def input_check(data, n_objects = None):
    """Check input data for correct type and dimensions

    Arguments
    ---------
    data : np.ndarray
        Array of arrays, each sub-array contains the binary object 
        OR Array with the columnwise sum, if so specify n_objects

    n_objects : int
        Number of objects, only necessary if the column wize sum is the input data."""

    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected input of type np.ndarray, got {type(data).__name__} instead.")

    
    if data.ndim == 1:
        c_total = data
        if not n_objects:
            raise ValueError("Input data is the columnwise sum, please specify number of objects")
    else:
        c_total = np.sum(data, axis = 0)
        if not n_objects:
            n_objects = len(data)      
        elif n_objects and n_objects != len(data):
            print("Warning, specified number of objects is different from the number of objects in data")
            n_objects = len(data)
            print("Doing calculations with", n_objects, "objects.")


    return c_total, n_objects