import pytest
import numpy as np

fps = np.load('tests/data/ECFP4_fps.npy')

def test_sigma():
    from iChem.iSIM.sigma import stratified_sigma, get_stdev_russell_fast
    from iChem.utils.utils import pairwise_average, rdkit_pairwise_sim

    # Test stratified sigma
    value = stratified_sigma(fps, n=50, n_ary="JT")
    assert value == pytest.approx(rdkit_pairwise_sim(fps, return_std=True)[1], abs=0.005)

    value = get_stdev_russell_fast(fps)
    assert value == pairwise_average(fps, return_std=True, n_ary='RR')[1]
    assert value == pytest.approx(stratified_sigma(fps, n=50, n_ary="RR"), abs = 0.005)