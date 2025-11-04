import iChem.iSIM.sampling as sampling
import iChem.iSIM as iSIM
import numpy as np # type: ignore
import pytest # type: ignore

# Load fingerprints for the test
fingerprints = np.load('tests/data/MACCS_fps.npy')
comp_sim = iSIM.calculate_comp_sim(fingerprints, n_ary = 'JT')

# Test medoid sampling
def test_medoid_sampling():
    medoids = sampling.medoid_sampling(fingerprints, n_ary = 'JT', n_sample=11)
    assert len(medoids) == 11
    assert np.sum([x in [16, 63, 14, 6, 12, 5, 11, 62, 40, 15, 99, 54] for x in medoids]) == 11

# Test outlier sampling
def test_outlier_sampling():
    outliers = sampling.outlier_sampling(fingerprints, n_ary = 'JT', n_sample = 11, comp_sim=comp_sim)
    assert len(outliers) == 11
    assert np.sum([x in [3, 56, 0, 78, 37, 34, 4, 51, 83, 32, 28, 80, 90, 79] for x in outliers]) == 11

# Test extremes sampling
def test_extremes_sampling():
    extremes = sampling.extremes_sampling(fingerprints, n_ary = 'JT', n_sample=10, comp_sim=comp_sim)
    assert np.sum([x in [16, 63, 14, 6, 12, 5, 83, 32, 28, 80, 90, 79] for x in extremes]) == 10

# Test stratified sampling
def test_stratified_sampling():
    stratified = sampling.stratified_sampling(fingerprints, n_ary = 'JT', n_sample= 11)
    assert len(stratified) == 11
    assert np.sum([x in [16, 54, 58, 115, 111, 94, 17, 9, 76, 44, 37] for x in stratified]) == 11

def test_quota_sampling():
    quota = sampling.quota_sampling(fingerprints, n_ary = 'JT', n_sample=11)
    assert len(quota) == 11
    assert np.sum([x in [16,  12,   8, 115, 118,  17, 103,   1,  56 ,  4,  90]for x in quota]) == 11

test_quota_sampling()