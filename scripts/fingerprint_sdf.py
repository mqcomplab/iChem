from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate fingerprints from an SDF file and output a .npy and the .smi')
parser.add_argument('-i', '--input', type=str, required=True, help='Input .sdf file')
parser.add_argument('-o', '--output', type=str, required=True, help='Output .npy file') 
parser.add_argument('-fpt', '--fp_type', type=str, default='RDKIT', choices=['RDKIT', 'ECFP4', 'ECFP6', 'MACCS'], help='Type of fingerprint to generate')
parser.add_argument('-n', '--n_bits',type=int, default=2048, help='Number of bits for the fingerprint (only for ECFP4 and ECFP6)')

args = parser.parse_args()

# Read molecules from the SDF file
suppl = Chem.SDMolSupplier(args.input)

# Get how many molecules were read
num_mols = len(suppl)

# Get the mol generator based on the fingerprint type
if args.fp_type == 'RDKIT':
    fp_generator = rdFingerprintGenerator.GetRDKitFPGenerator()
elif args.fp_type == 'ECFP4':
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=args.n_bits)
elif args.fp_type == 'ECFP6':
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=args.n_bits)
elif args.fp_type == 'MACCS':
    class fp_generator():
        @staticmethod
        def GetFingerprintAsNumPy(mol):
            return np.array(MACCSkeys.GenMACCSKeys(mol))
            
else:
    raise ValueError('Invalid fingerprint type: {}'.format(args.fp_type))

fingerprints = []
smiles = []
for mol in suppl:
    if mol is None:
        continue
    else:
        try:
            fp = fp_generator.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
            smiles.append(Chem.MolToSmiles(mol))
        except:
            print('Error generating fingerprint for molecule')
            continue

# Save the fingerprints to a .npy file
fingerprints = np.array(fingerprints)
np.save(args.output, fingerprints)
print(f'{len(fingerprints)} fingerprints saved to {args.output}')

# Save the SMILES to a .smi file
with open(args.output.replace('.npy', '.smi'), 'w') as f:
    for smi in smiles:
        f.write(smi + '\n')
print(f'{len(smiles)} SMILES saved to {args.output.replace(".npy", ".smi")}')



