import iChem.utils as utils
import iChem.iSIM as iSIM
from rdkit import DataStructs # type: ignore
import pytest # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

fps = np.load('data/RDKIT_fps.npy')

# Test the npy_to_rdkit function
def test_npy_to_rdkit():
    # Convert numpy array to RDKit fingerprints
    fp_rdkit = utils.npy_to_rdkit(fps)
    assert type(fp_rdkit[0]) == DataStructs.cDataStructs.ExplicitBitVect

# Test the rdkit_pairwise_sim function
def test_rdkit_pairwise_sim():
    # Calculate the pairwise similarity
    value = utils.rdkit_pairwise_sim(fps)
    assert value == pytest.approx(0.198477)
    assert value == utils.pairwise_average(fps, n_ary="JT")

# Test the rdkit_pairwise_matrix function
def test_rdkit_pairwise_matrix():
    # Calculate the pairwise matrix
    value = utils.rdkit_pairwise_matrix(fps)

    dimensions = value.shape
    assert dimensions == (119, 119)
    assert value.diagonal().sum() == 119
    assert value[0, 1] == iSIM.calculate_isim(np.array([fps[0], fps[1]]), n_ary="JT")

# Test the pairwise_average function
def test_pairwise_average():
    # Calculate the average similarity
    value = utils.pairwise_average(fps, n_ary="JT")
    assert value == pytest.approx(0.198477)

# Test the pairwise avrage for real fingerprints
def test_pairwise_average_real():
    # Calculate the average similarity for real fingerprints
    smiles = pd.read_csv('data/logP_data.csv')
    smiles = smiles['SMILES']
    fps = utils.real_fps(smiles)
    fps = utils.minmax_norm(fps)
    value = utils.pairwise_average_real(fps, n_ary="RR")
    assert value == pytest.approx(0.071182)

    value = utils.pairwise_average_real(fps, n_ary="SM")
    assert value == pytest.approx(0.7252999199)

    jt_value = utils.pairwise_average_real(fps, n_ary="JT")
    assert jt_value == pytest.approx(0.47739, abs = 0.05)

# Test the real_fps function and normalization
def test_real_fps():
    # Calculate the real fingerprints
    smiles = pd.read_csv('data/logP_data.csv')
    smiles = smiles['SMILES']
    fps, invalid_smiles = utils.real_fps(smiles, return_invalid = True)
    assert type(fps) == np.ndarray
    assert fps.shape[0] + len(invalid_smiles) == 119
    assert np.nan not in fps

    # Calculate the minmax normalization
    value_norm = utils.minmax_norm(fps)
    assert value_norm.min() == 0
    assert value_norm.max() == 1
    assert value_norm.shape