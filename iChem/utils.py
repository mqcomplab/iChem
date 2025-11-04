import numpy as np # type: ignore
import pandas as pd # type: ignore
from .iSIM import calculate_isim
from .iSIM.real import pair_jt, pair_rr, pair_sm
from rdkit import Chem, DataStructs # type: ignore
from rdkit.Chem import Descriptors, rdFingerprintGenerator, MACCSkeys # type: ignore

"""
This module contains utility functions for the iChem package regarding fingerprint generation, and 
pairwise similarity calculations using RDKit functions.
"""
def binary_fps(smiles: list, fp_type: str = 'RDKIT', n_bits: int = 2048, return_invalid: bool = True) -> np.ndarray:
    """
    This function generates binary fingerprints for the dataset.
    
    Parameters:
    smiles: list of SMILES strings
    fp_type: type of fingerprint to generate ['RDKIT', 'ECFP4', 'ECFP6', or 'MACCS']
    n_bits: number of bits for the fingerprint
    return_invalid: whether to return invalid SMILES indices
    
    Returns:
    fingerprints: numpy array of fingerprints
    and list of invalid SMILES indices if return_invalid is True
    """
    # Generate the fingerprints
    if fp_type == 'RDKIT':
        fps_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5, fpSize=n_bits)
    elif fp_type == 'ECFP4':
        fps_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    elif fp_type == 'ECFP6':
        fps_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
    elif fp_type == 'MACCS':
        class fps_gen:
            def GetFingerprintAsNumPy(mol):
                fp = np.zeros((167,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), fp)
                return fp
    else:
        print('Invalid fingerprint type: ', fp_type)
        exit(0)

    fingerprints = []
    invalid_smiles = []
    for k, smi in enumerate(smiles):
        # Generate the mol object
        try:
          mol = Chem.MolFromSmiles(smi)
        except:
          print('Invalid SMILES: ', smi)
          invalid_smiles.append(k)
          exit(0)

        try:
            # Generate the fingerprint and append to the list
            fingerprint = np.array([], dtype=np.uint8)
            fingerprint = fps_gen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fingerprint)
        except:
            print('Error generating fingerprint for SMILES: ', smi)
            invalid_smiles.append(k)

    fingerprints = np.array(fingerprints)
    if return_invalid:
        return fingerprints, invalid_smiles
    else:
        return fingerprints

def real_fps(smiles, return_invalid: bool = False):
    """
    This function generates real number fingerprints for the dataset based on RDKit descriptors.
    Skips corrupted smiles strings. 
    
    Parameters:
    smiles: list of SMILES strings
    
    Returns:
    fingerprints: numpy array of fingerprints
    """
    fps = []
    invalid_smiles = []
    for k, smi in enumerate(smiles):
        # Generate the mol object
        try:
            mol = Chem.MolFromSmiles(smi)
            try:
                des = []
                for nm, fn in Descriptors._descList:
                    val = fn(mol)
                    des.append(val)
                fps.append(des)
            except:
                print('Error computing descriptor: ', nm)
                invalid_smiles.append(k)
                continue             
        except:
            print('Invalid SMILES: ', smi)
            invalid_smiles.append(k)

    # Convert to numpy array
    fps = np.array(fps)
    if return_invalid:
        return fps, invalid_smiles
    else:
        return fps

def minmax_norm(fps):
    """
    This function performs min-max normalization on the dataset. Required for the calculation of iSIM for real
    number fingerprints. Eliminates columns with NaN values or where all values are zero.

    Parameters:
    fps: numpy array of fingerprints

    Returns:
    fps: normalized numpy array of fingerprints
    """

    # Turn the array into a DataFrame
    df = pd.DataFrame(fps)

    # Normalize the data
    df_numeric = df.select_dtypes(include = [np.number])
    columns = df_numeric.columns

    for column in columns:
        min_prop = np.min(df[column])
        max_prop = np.max(df[column])

        if min_prop == max_prop:
            df = df.drop(column, axis = 1)
            continue

        try:
            df[column] = [(x - min_prop) / (max_prop - min_prop) for x in df[column]]
        except ZeroDivisionError:
            df.drop(column, axis = 1)

    df = df.dropna(axis = 'columns')

    # Return the normalized data as a numpy array
    return df.to_numpy()

def npy_to_rdkit(fps_np):
    """
    This function converts numpy array fingerprints to RDKit fingerprints.

    Parameters:
    fps_np: numpy array of fingerprints

    Returns:
    fp_rdkit: list of RDKit fingerprints
    """
    fp_len = len(fps_np[0])
    fp_rdkit = []
    for fp in fps_np:
        bitvect = DataStructs.ExplicitBitVect(fp_len)
        bitvect.SetBitsFromList(np.where(fp)[0].tolist())
        fp_rdkit.append(bitvect)
    
    return fp_rdkit


def rdkit_pairwise_sim(fingerprints, return_std: bool = False):
    """
    This function computes the pairwise similarity between all objects in the dataset using Jaccard-Tanimoto similarity.

    Parameters:
    fingerprints: list of fingerprints

    Returns:
    similarity: average similarity between all objects 
    and standard deviation if return_std is True
    """
    if type(fingerprints[0]) == np.ndarray:
        fingerprints = npy_to_rdkit(fingerprints)

    nfps = len(fingerprints)
    similarity = []

    for n in range(nfps - 1):
        sim = DataStructs.BulkTanimotoSimilarity(fingerprints[n], fingerprints[n+1:])
        similarity.extend([s for s in sim])

    if return_std:
        return np.mean(similarity), np.std(similarity)
    else:
        return np.mean(similarity)


def rdkit_pairwise_matrix(fingerprints):
    """
    This function computes the pairwise similarity between all objects in the dataset using Jaccard-Tanimoto similarity.

    Parameters:
    fingerprints: list of fingerprints

    Returns:
    similarity: matrix of similarity values
    """
    if type(fingerprints[0]) == np.ndarray:
        fingerprints = npy_to_rdkit(fingerprints)

    n = len(fingerprints)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1)  # Set diagonal values to 1

    # Fill the upper triangle directly while computing similarities
    for i in range(n - 1):
        sim = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1:])
        for j, s in enumerate(sim):
            matrix[i, i + 1 + j] = s  # Map similarities to the correct indices in the upper triangle

    return matrix
  
def pairwise_average(fingerprints: np.ndarray, n_ary: str = 'JT', return_std: bool = False):
    """
    This function computes the pairwise average similarity between all objects in the dataset.
    
    Parameters:
    fingerprints: numpy array of fingerprints
    n_ary: type of similarity to index compute
    
    Returns:
    average: average similarity between all objects
    and standard deviation if return_std is True
    """
   
    # Compute the pairwise similarities
    pairwise_sims = []
    for i in range(len(fingerprints) - 1):
        for j in range(i + 1, len(fingerprints)):
            pairwise_sims.append(calculate_isim(np.array([fingerprints[i], fingerprints[j]]), n_ary = n_ary))

    # Compute the average similarity
    average = np.mean(pairwise_sims)
    
    if return_std:
        return average, np.std(pairwise_sims)
    else:
        return average
    
def pairwise_average_real(fingerprints: np.ndarray, n_ary: str = 'JT', return_std: bool = False):
    """
    This function computes the pairwise average similarity between all objects in the dataset.
    
    Parameters:
    fingerprints: numpy array of fingerprints
    n_ary: type of similarity to index compute
    
    Returns:
    average: average similarity between all objects
    and standard deviation if return_std is True
    """
    # Normalize the fingerprints
    normalized_fps = minmax_norm(fingerprints)

    # Define the comparison function
    if n_ary == 'JT':
        compare = pair_jt
    elif n_ary == 'RR':
        compare = pair_rr
    elif n_ary == 'SM':
        compare = pair_sm
    else:
        print('Invalid similarity index: ', n_ary)
        exit(0)
    
    # Compute the pairwise similarities
    pairwise_sims = []
    for i in range(len(normalized_fps) - 1):
        for j in range(i + 1, len(normalized_fps)):
            pairwise_sims.append(compare(normalized_fps[i], normalized_fps[j]))

    # Compute the average similarity
    average = np.mean(pairwise_sims)
    
    if return_std:
        return average, np.std(pairwise_sims)
    else:
        return average