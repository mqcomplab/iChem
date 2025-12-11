from iChem.libchem import LibChem, LibComparison
from iChem.bblean import pack_fingerprints, unpack_fingerprints
import numpy as np # type: ignore
import pytest # type: ignore
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

