iChem - Instant Cheminformatics

<img src="img/iChem.png" alt="iChem Logo" width="75%">

Cheminformatics package by the Miranda-Quintana group at the University of Florida

iChem compares binary fingerprints in sets simultaneously instead of pairwise.

## Installation

iChem requires Python 3.12 or newer.

At the moment, iChem is installed from this repository and not from PyPI.
It depends on the upstream `bblean` package, which supports Python 3.11+; there is no
Python-version conflict with iChem's 3.12+ requirement.

### Create a new conda environment (recommended)
```bash
conda create -n iChem python=3.12 -y
conda activate iChem
```

### Install RDKit (required)
```bash
conda install -c conda-forge rdkit
```

### Clone the repository
```bash
git clone https://github.com/klopezperez/iChem.git
cd iChem
```

### Install iChem from source
```bash
BITBIRCH_BUILD_CPP=1 pip install -e .
```

This installs iChem together with `bblean` from the upstream repository and builds the
C++ extension during installation.

## Features

- **Fast Similarity Calculations**: Compute similarity metrics for entire chemical libraries simultaneously
- **Clustering Tools**: Efficient clustering algorithms for chemical libraries using BitBIRCH
- **Library Comparison**: Compare multiple chemical libraries using various methodologies (intraiSIM, interiSIM)
- **Visualization**: Built-in plotting tools for cluster composition, heatmaps, and molecular visualizations
- **Flexible Fingerprints**: Support for multiple fingerprint types (ECFP4, MACCS, RDKit)

## Submodules

- **`iChem.iSIM`**: Core n-ary similarity tools, including instantaneous similarity calculations, complementary similarity, medoid and outlier identification, counters, sampling, and sigma-based analyses.
- **`iChem.bblean`**: Memory-efficient BitBIRCH-style clustering for binary molecular fingerprints, along with fingerprint packing, SMILES loading, similarity utilities, and a `hierarchical` workflow for multi-level binary clustering. This code now relies on the upstream `bblean` package, so `BITBIRCH_BUILD_CPP=1 pip install -e .` will build the C++ extension during an editable install.
- **`iChem.bbreal`**: Clustering tools for real-valued descriptor spaces, including threshold estimation and a `hierarchical` workflow for multi-level clustering of continuous molecular representations.
- **`iChem.libchem`**: High-level library analysis interfaces built around `LibChem` and `LibComparison` for loading libraries, generating fingerprints, clustering, and comparing multiple collections.
- **`iChem.visualization`**: Plotting and visualization helpers for cluster populations, heatmaps, cluster connectivity, and molecule image generation.
- **`iChem.utils`**: General utility functions for SMILES loading, fingerprint generation, normalization, and pairwise similarity calculations using RDKit.

The [scripts/](scripts/) directory contains notebooks and example scripts showing common iChem workflows.

### Notebooks

- `bbreal_example.ipynb`: Demonstrates `iChem.bbreal` usage with a sample dataset and clustering examples.
- `hierarchical_bitbirch.ipynb`: Visualizes cluster connectivity and cluster-level summaries for inspection and publication-ready figures. Hierarchical visualizations with bblean.
- `iSIM_start_guide.ipynb`: A quickstart guide for `iChem.iSIM`, showing typical workflows and basic analyses.
- `library_compare.ipynb`: Example workflow for comparing multiple chemical libraries using iChem's comparison tools.

Other files in `scripts/` are example Python scripts for batch processing, fingerprint generation, and format conversion.


## Citation

If you use iChem in your research, please cite our work.

