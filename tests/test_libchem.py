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

    lib.set_optimal_threshold()
    assert lib.optimal_threshold == isim + 3.5 * isim_sigma

    lib.set_optimal_threshold(0.67)
    assert lib.optimal_threshold == 0.67