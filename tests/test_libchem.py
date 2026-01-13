from iChem.libchem import LibChem, LibComparison
from iChem.bblean import pack_fingerprints, unpack_fingerprints
import numpy as np # type: ignore
import pytest # type: ignore
from unittest.mock import patch
from PIL import Image
np.random.seed(42)

def test_load_smiles():
    lib = LibChem()
    lib.load_smiles('tests/data/molecules.smi')
    assert lib.n_molecules == 119

def test_generate_fingerprints():
    lib = LibChem()
    lib.load_smiles('tests/data/molecules.smi')
    lib.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
    assert lib.fps_packed.shape == (118, 256)
    assert lib.n_molecules == 118

def test_load_fingerprints():
    lib = LibChem()
    data = np.load('tests/data/RDKIT_fps.npy')
    data = data.astype(np.uint8)
    lib.load_fingerprints(data, packed=False)
    assert lib.fps_packed.shape == (119, 256)
    assert lib.n_molecules == 119
    assert unpack_fingerprints(lib.fps_packed).shape == (119, 2048)

    with pytest.raises(ValueError):
        lib.load_smiles('tests/data/molecules.smi')

def test_get_fingeprints():
    lib = LibChem()
    lib.load_smiles('tests/data/molecules.smi')
    lib.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
    fps = lib.get_fingerprints()
    assert fps.shape == (118, 256)
    fps = lib.get_fingerprints(packed=False)
    assert fps.shape == (118, 2048)

def test_get_iSIM():
    lib = LibChem()
    lib.load_smiles('tests/data/molecules.smi')
    lib.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
    isim = lib.get_iSIM()
    assert isim == pytest.approx(0.1128, rel=1e-4)
    isim_sigma = lib.get_iSIM_sigma()
    assert isinstance(isim_sigma, float)
    assert isim_sigma == pytest.approx(0.07183, rel=1e-4)

    lib.set_threshold()
    assert lib.threshold == isim + 3.5 * isim_sigma

    lib.set_threshold(0.1128)
    assert lib.threshold == 0.1128

    medoids = lib.get_cluster_medoids(return_smiles=False)
    assert len(medoids) == 5

    medoids, smiles = lib.get_cluster_medoids(return_smiles=True)

    assert smiles == ['C1=CC=C(C=C1)CC(C(=O)[O-])NC(=O)C(CC2=CNC3=CC=CC=C32)N',
                      'CC(C)[NH2+]CC(O)COc1cccc2ccccc12',
                      'CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)[O-])C',
                      '[NH3+]CCCCCc1ccccn1',
                      'O=C1C[NH+]=C(c2ccccc2)c2cc([N+](=O)[O-])ccc2N1']
    
def test_lib_comparison():
    lib1 = LibChem()
    lib1.load_smiles('tests/data/molecules.smi')
    lib1.generate_fingerprints(fp_type='ECFP4', n_bits=2048)

    lib2 = LibChem()
    lib2.load_smiles('tests/data/molecules.smi')
    lib2.generate_fingerprints(fp_type='ECFP4', n_bits=2048)

    libcomp = LibComparison()
    libcomp.add_library(lib1, 'Library 1')
    libcomp.add_library(lib2, 'Library 2')
    value = libcomp.compare_medoids(methodology='MaxSum')
    assert value == pytest.approx(1.0, rel=1e-4)
    value = libcomp.compare_libraries(methodology='intraiSIM')
    assert lib1.get_iSIM() == lib2.get_iSIM()

    lib3 = LibChem()
    lib3.load_smiles('tests/data/mcule_natural_products.smi')
    lib3.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
    libcomp.add_library(lib3, 'Library 3')

    results = libcomp.compare_medoids_all(methodology='MaxSum')
    assert results[0][0] == results[0][1]
    assert results[0][1] > results[0][2]

    counts, _ = libcomp.cluster_classification_counts()
    assert counts['Library 1+Library 2'] > counts['Library 2+Library 3']
    assert counts['Library 3'] > counts['Library 1'] + counts['Library 2'] + counts['Library 1+Library 2']

   
# Test plot_cluster_composition
def test_cluster_visualization_and_composition():
    """Test cluster visualization and composition plotting methods"""
    lib1 = LibChem()
    lib1.load_smiles('tests/data/molecules.smi')
    lib1.generate_fingerprints(fp_type='ECFP4', n_bits=2048)

    lib2 = LibChem()
    lib2.load_smiles('tests/data/molecules.smi')
    lib2.generate_fingerprints(fp_type='ECFP4', n_bits=2048)

    libcomp = LibComparison()
    libcomp.add_library(lib1, 'Library 1')
    libcomp.add_library(lib2, 'Library 2')
    
    # Cluster the libraries once
    libcomp.cluster_libraries(methodology='medoids')
    
    with patch('matplotlib.pyplot.show'):
        a = libcomp.plot_cluster_composition(
            lib_names=['Library 1', 'Library 2'],
            top=5
        )
        assert a is None  # plt.show() returns None
    
    # Test cluster_visualization returns a PIL Image
    img = libcomp.cluster_visualization(cluster_number=10)
    assert img is not None
    assert isinstance(img, Image.Image)

def test_get_cluster_samples():
    """Test stratified sampling from clusters"""
    lib = LibChem()
    lib.load_smiles('tests/data/molecules.smi')
    lib.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
    
    sampled_fps = lib.get_cluster_samples(n_samples=50)
    assert sampled_fps.shape[0] == 50
    
    sampled_fps, smiles = lib.get_cluster_samples(n_samples=50, return_smiles=True)
    assert len(smiles) == 50
    assert sampled_fps.shape[0] == 50
    
    sampled_fps, cluster_ids = lib.get_cluster_samples(n_samples=50, return_cluster_ids=True)
    assert len(cluster_ids) == 50
    assert sampled_fps.shape[0] == 50

def test_get_cluster_flags():
    """Test retrieving flags for each cluster"""
    lib = LibChem()
    lib.load_smiles('tests/data/molecules.smi')
    lib.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
    lib.load_flags(['flag_' + str(i) for i in range(118)])
    
    cluster_flags = lib.get_cluster_flags()
    assert isinstance(cluster_flags, list)
    assert all(isinstance(cf, list) for cf in cluster_flags)

def test_load_flags_validation():
    """Test flag loading validation"""
    lib = LibChem()
    lib.load_smiles('tests/data/molecules.smi')
    
    with pytest.raises(ValueError):
        lib.load_flags(['flag1', 'flag2'])  # Mismatch with n_molecules

def test_cluster_with_custom_threshold():
    """Test clustering with custom threshold"""
    lib = LibChem()
    lib.load_smiles('tests/data/molecules.smi')
    lib.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
    
    lib.cluster(threshold=0.5)
    assert lib.threshold == 0.5
    assert hasattr(lib, 'clusters')
    assert isinstance(lib.clusters, list)