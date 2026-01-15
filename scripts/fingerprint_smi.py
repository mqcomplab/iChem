from rdkit import Chem # type: ignore
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator # type: ignore
import numpy as np # type: ignore
import argparse

parser = argparse.ArgumentParser(description='Generate fingerprints from an SDF file and output a .npy')
parser.add_argument('-i', '--input', type=str, required=True, help='Input .smi file')
parser.add_argument('-o', '--output', type=str, required=True, help='Output .npy file') 
parser.add_argument('-fpt', '--fp_type', type=str, default='RDKIT', choices=['RDKIT', 'ECFP4', 'ECFP6', 'MACCS'], help='Type of fingerprint to generate')
parser.add_argument('-n', '--n_bits',type=int, default=2048, help='Number of bits for the fingerprint (only for ECFP4 and ECFP6)')

args = parser.parse_args()

# Read the smiles from the .smi file
smiles = []
with open(args.input, 'r') as f:
    for line in f:
        smiles.append(line.strip().split()[0])

# Get how many molecules were read
num_mols = len(smiles)

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
valid_smiles = []
for smi in smiles:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print('Error generating molecule from SMILES: {}'.format(smi))
            continue
        fp = fp_generator.GetFingerprintAsNumPy(mol)
        fingerprints.append(fp)
        valid_smiles.append(smi)
    except:
        print('Error generating fingerprint for molecule')
        continue

# Rewrite the input .smi file with only valid SMILES
with open(args.input, 'w') as f:
    for smi in valid_smiles:
        f.write(smi + '\n')

# Save the fingerprints to a .npy file
fingerprints = np.array(fingerprints)
np.save(args.output, fingerprints)
print(f'{len(fingerprints)} fingerprints saved to {args.output}')