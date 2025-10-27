import numpy as np
import unittest
from iChem.iSIM.real import pair_jt, pair_rr, pair_sm, process_matrix, calculate_isim_real
from iChem.utils import minmax_norm, real_fps
import pandas as pd

class TestPairwiseSimilarityFunctions(unittest.TestCase):

    def setUp(self):
        # Sample fingerprints for testing
        self.fp1 = np.array([0.1, 0.2, 0.3, 0.4])
        self.fp2 = np.array([0.5, 0.5, 0.5, 0.5])  # Edge case: uniform values
        self.normalized_fps = minmax_norm(np.array([self.fp1, self.fp2]))

        self.fp1 = self.normalized_fps[0]
        self.fp2 = self.normalized_fps[1]

    def test_pair_jt(self):
        # Test Jaccard-Tanimoto similarity
        result = pair_jt(self.fp1, self.fp2)
        expected = np.dot(self.fp1, self.fp2) / (
            np.dot(self.fp1, self.fp1) + np.dot(self.fp2, self.fp2) - np.dot(self.fp1, self.fp2)
        )
        self.assertAlmostEqual(result, expected, places=6)

    def test_pair_rr(self):
        # Test Real-Real similarity
        result = pair_rr(self.fp1, self.fp2)
        m = len(self.fp1)
        expected = np.dot(self.fp1, self.fp2) / m
        self.assertAlmostEqual(result, expected, places=6)

        # Test with custom m
        custom_m = 10
        result_custom_m = pair_rr(self.fp1, self.fp2, m=custom_m)
        expected_custom_m = np.dot(self.fp1, self.fp2) / custom_m
        self.assertAlmostEqual(result_custom_m, expected_custom_m, places=6)

    def test_pair_sm(self):
        # Test Similarity Measure
        result = pair_sm(self.fp1, self.fp2)
        m = len(self.fp1)
        expected = (np.dot(self.fp1, self.fp2) + np.dot(1 - self.fp1, 1 - self.fp2)) / m
        self.assertAlmostEqual(result, expected, places=6)

    def test_edge_case_uniform_fingerprints(self):
        # Test edge case with uniform fingerprints
        result = pair_jt(self.fp2, self.fp2)
        self.assertAlmostEqual(result, 1.0, places=6)

        result = pair_rr(self.fp2, self.fp2)
        self.assertAlmostEqual(result, 1.0, places=6)

        result = pair_sm(self.fp2, self.fp2, m = None)
        self.assertAlmostEqual(result, 1.0, places=6)

class TestRealISIMFunctions(unittest.TestCase):

    def setUp(self):
        # Sample fingerprints for testing
        self.fingerprints = minmax_norm(real_fps(pd.read_csv('tests/data/logP_data.csv')['SMILES'].values))

    def test_calculate_isim_real(self):
        # Test the calculate_isim_real function
        iSIM_jt = calculate_isim_real(self.fingerprints, n_ary='JT')
        iSIM_rr = calculate_isim_real(self.fingerprints, n_ary='RR')
        iSIM_sm = calculate_isim_real(self.fingerprints, n_ary='SM')

        expected_iSIM_jt = 0.477398141416
        expected_iSIM_rr = 0.071182
        expected_iSIM_sm = 0.7252999199

        self.assertAlmostEqual(iSIM_jt, expected_iSIM_jt, places=6)
        self.assertAlmostEqual(iSIM_rr, expected_iSIM_rr, places=6)
        self.assertAlmostEqual(iSIM_sm, expected_iSIM_sm, places=6)

if __name__ == '__main__':
    unittest.main()