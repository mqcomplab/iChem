"""                             iSIM_MODULES
    ----------------------------------------------------------------------
    
    Miranda-Quintana Group, Department of Chemistry, University of Florida 
    
    ----------------------------------------------------------------------
    
    Please, cite the original paper on iSIM:

    https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00041b

    """

import numpy as np

from .counters import calculate_counters
from ._input_check import input_check

__all__ = ['calculate_isim', 
           'calculate_comp_sim',
           'calculate_outlier', 
           'calculate_medoid']

def isim_jt(data, n_objects = None) -> float:
    """Calculate the iSIM index for JT

    Arguments
    ---------
    data : np.ndarray
        Array of arrays, each sub-array contains the binary object 
        OR Array with the columnwise sum, if so specify n_objects

    n_objects : int
        Number of objects, only necessary if the column wize sum is the input data."""

    c_total, n_objects = input_check(data = data, n_objects = n_objects)

    sum_kq = np.sum(c_total)
    sum_kqsq = np.dot(c_total, c_total)
    a = (sum_kqsq - sum_kq)/2

    return a/(a + n_objects * sum_kq - sum_kqsq)

def isim_rr(data, n_objects = None) -> float:
    """Calculate the iSIM index for RR

    Arguments
    ---------
    data : np.ndarray
        Array of arrays, each sub-array contains the binary object 
        OR Array with the columnwise sum, if so specify n_objects

    n_objects : int
        Number of objects, only necessary if the column wize sum is the input data."""

    c_total, n_objects = input_check(data = data, n_objects = n_objects)

    a = np.sum(c_total * (c_total - 1) / 2)
    p = n_objects * (n_objects - 1) * len(c_total) / 2

    return a/p

def isim_sm(data, n_objects = None) -> float:
    """Calculate the iSIM index for SM

    Arguments
    ---------
    data : np.ndarray
        Array of arrays, each sub-array contains the binary object 
        OR Array with the column wize sum, if so specify n_objects

    n_objects : int
        Number of objects, only necessary if the column wize sum is the input data."""

    c_total, n_objects = input_check(data = data, n_objects = n_objects)

    a = np.sum(c_total * (c_total - 1) / 2)
    off_coincidences = n_objects - c_total
    d = np.sum(off_coincidences * (off_coincidences - 1) / 2)
    p = n_objects * (n_objects - 1) * len(c_total) / 2

    return (a + d)/p

def calculate_isim(data, n_objects = None, n_ary = 'JT') -> float:
    """Calculate the iSIM index for RR, JT, or SM

    Arguments
    ---------
    data : np.ndarray
        Array of arrays, each sub-array contains the binary object 
        OR Array with the columnwise sum, if so specify n_objects
    
    n_objects : int
        Number of objects, only necessary if the column wize sum is the input data.

    n_ary : str
        String with the initials of the desired similarity index to calculate the iSIM from. 
        Only RR, JT, or SM are available. For other indexes use gen_sim_dict.

    Returns
    -------
    isim : float
        iSIM index for the specified similarity index.
        iSIM corresponds to the average similarity of the set calculated linearly.
    """

    # Calculate only necessary counters for the desired index 

    if n_ary == 'RR':
        return isim_rr(data = data, n_objects = n_objects)
    
    elif n_ary == 'JT':
        return isim_jt(data = data, n_objects = n_objects)
    
    elif n_ary == 'SM':
        return isim_sm(data = data, n_objects = n_objects)


def gen_sim_dict(data, n_objects = None) -> dict:
    """Calculate a dictionary containing all the available similarity indexes

    Arguments
    ---------
    See calculate counters.

    Returns
    -------
    sim_dict : dict
        Dictionary with all the available iSIM similarity indexes in the module."""
 
    # Indices
    # AC: Austin-Colwell, BUB: Baroni-Urbani-Buser, CTn: Consoni-Todschini n
    # Fai: Faith, Gle: Gleason, Ja: Jaccard, Ja0: Jaccard 0-variant
    # JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
    # SM: Sokal-Michener, SSn: Sokal-Sneath n
 
    # Calculate the similarity and dissimilarity counters
    counters = calculate_counters(data = data, n_objects = n_objects)

    ac = (2/np.pi) * np.arcsin(np.sqrt(counters['total_sim']/
                                       counters['p']))
    bub = ((counters['a'] * counters['d'])**0.5 + counters['a'])/\
          ((counters['a'] * counters['d'])**0.5 + counters['a'] + counters['total_dis'])
    fai = (counters['a'] + 0.5 * counters['d'])/\
          (counters['p'])
    gle = (2 * counters['a'])/\
          (2 * counters['a'] + counters['total_dis'])
    ja = (3 * counters['a'])/\
         (3 * counters['a'] + counters['total_dis'])
    jt = (counters['a'])/\
         (counters['a'] + counters['total_dis'])
    rt = (counters['total_sim'])/\
         (counters['p'] + counters['total_dis'])
    rr = (counters['a'])/\
         (counters['p'])
    sm = (counters['total_sim'])/\
         (counters['p'])
    ss1 = (counters['a'])/\
          (counters['a'] + 2 * counters['total_dis'])
    ss2 = (2 * counters['total_sim'])/\
          (counters['p'] + counters['total_sim'])

    # Dictionary with all the results
    Indices = {'AC': ac, 'BUB':bub, 'Fai':fai, 'Gle':gle, 'Ja':ja,
               'JT':jt, 'RT':rt, 'RR':rr, 'SM':sm, 'SS1':ss1, 'SS2':ss2}
    #Indices = {'Fai':fai, 'Gle':gle, 'Ja':ja,
    #           'JT':jt, 'RT':rt, 'RR':rr, 'SM':sm, 'SS1':ss1, 'SS2':ss2}
    return Indices