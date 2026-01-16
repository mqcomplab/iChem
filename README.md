iChem - Instant Cheminformatics

<img src="img/iChem.png" alt="iChem Logo" width="75%">

Cheminformatics package by the Miranda-Quintana group at the University of Florida

iChem compares binary fingerprints in sets simultaneously instead of pairwise.

## Installation

### Create a new conda environment (recommended)
```bash
conda create -n iChem python=3.10 -y
conda activate iChem
```

### Install RDKit (required)
```bash
conda install -c conda-forge rdkit
```

### Install iChem
```bash
pip install iChem
```

## Features

- **Fast Similarity Calculations**: Compute similarity metrics for entire chemical libraries simultaneously
- **Clustering Tools**: Efficient clustering algorithms for chemical libraries using BitBIRCH
- **Library Comparison**: Compare multiple chemical libraries using various methodologies (intraiSIM, interiSIM)
- **Visualization**: Built-in plotting tools for cluster composition, heatmaps, and molecular visualizations
- **Flexible Fingerprints**: Support for multiple fingerprint types (ECFP4, MACCS, RDKit)

## Getting Started

The `scripts/` directory contains Jupyter notebooks and Python scripts that serve as comprehensive guides for using iChem's various tools:

- **`library_compare.ipynb`**: Tutorial for comparing chemical libraries, including similarity metrics and visualization
- **`fingerprint_smi.py`**: Script for generating fingerprints from SMILES files
- **`fingerprint_sdf.py`**: Script for generating fingerprints from SDF files

These examples demonstrate real-world usage patterns and best practices for working with iChem.


## Citation

If you use iChem in your research, please cite our work.

