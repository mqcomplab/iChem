from iChem.libchem import LibChem
import numpy as np # type: ignore

MyLib = LibChem()
fps = np.load('data/ECFP4_fps.npy')
fps = fps.astype(np.uint8)
MyLib.load_fingerprints(fps, packed=False)
print(MyLib.get_iSIM())
print(MyLib.get_iSIM_sigma(n_sigma_samples=10))