#!/usr/bin/env python3
"""
Convert SDF file to SMILES file (.smi).
Extracts SMILES strings from molecules and writes to output file (no header).
"""

import argparse
from rdkit import Chem

def sdf_to_smi(sdf_file, output_file):
    """
    Convert SDF file to SMILES file.
    
    Parameters:
    sdf_file: path to input SDF file
    output_file: path to output SMILES file
    """
    supplier = Chem.SDMolSupplier(sdf_file)
    
    valid_count = 0
    invalid_count = 0
    
    with open(output_file, 'w') as f:
        for mol in supplier:
            if mol is None:
                invalid_count += 1
                continue
            
            try:
                smiles = Chem.MolToSmiles(mol)
                f.write(smiles + '\n')
                valid_count += 1
            except Exception as e:
                print(f"Error converting molecule: {e}")
                invalid_count += 1
    
    print(f"Conversion complete:")
    print(f"  Valid SMILES written: {valid_count}")
    print(f"  Invalid/skipped: {invalid_count}")
    print(f"  Output file: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SDF file to SMILES file')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input SDF file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output SMILES file (default: input_file.smi)')
    
    args = parser.parse_args()
    
    output_file = args.output if args.output else args.input.rsplit('.', 1)[0] + '.smi'
    
    sdf_to_smi(args.input, output_file)
