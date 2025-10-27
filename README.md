iChem - Instanst Cheminformatics

<img src="img/iChem.png" alt="iChem Logo" width="75%">

Cheminformatics package by the Miranda-Quintana group at the University of Florida

iChem compares binary fingerprints in sets simultaneously instead of pairwise.

# Create a new conda environment (recommended)
conda create -n iChem python=3.10 -y
conda activate iChem

# Install RDKit (required)
conda install -c conda-forge rdkit

# Install iChem
pip install iChem