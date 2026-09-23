"""Command line entry points for iChem."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence
import pickle as pkl
import sys

from .bitbirch.cluster import cluster
from .bitbirch.multiround_reclustering import run_multiround_reclustering
from .bitbirch._hpc_initial_submit import prepare_initial_round_jobs
from .bitbirch._hpc_midsection_submit import prepare_midsection_round_jobs
from .bitbirch._hpc_final_submit import prepare_final_round_job
from .bitbirch import _config
from .utils.fingerprints import binary_fps, count_fps, real_fps
from .cluster_sampling import sample_clusters
from .cluster_sampling.sample_from_cluster_files import sample_from_cluster_files
from .cluster_analysis.rewrite_smiles import (
    rewrite_smiles_by_cluster,
    rewrite_smiles_by_cluster_from_npy_dir,
    rewrite_single_cluster_from_npy,
    prepare_rewrite_smiles_npy_jobs,
    find_first_missing_cluster,
    find_first_missing_cluster_from_files,
)
from ._cli import load_smiles, get_smi_files, get_output_path
import numpy as np


def _print_banner() -> None:
    """Print iChem banner with group attribution."""
    banner = r"""
      _     _     _     _     _
     / \   / \   / \   / \   / \
    |010|-|101|-|010|-|101|-|010|
     \_/   \_/   \_/   \_/   \_/
    
    iChem: Instant Cheminformatics
      _     _     _     _     _    
     / \   / \   / \   / \   / \
    |101|-|010|-|101|-|010|-|101|
     \_/   \_/   \_/   \_/   \_/

    Miranda-Quintana Group
    University of Florida
    Department of Chemistry
    """
    print(banner)


def _path_list(value: str) -> list[Path]:
    path = Path(value)
    if path.is_dir():
        return sorted(path.glob("*.npy"))
    return [path]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iChem", description="iChem BitBIRCH tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cluster_parser = subparsers.add_parser("cluster", help="Cluster a file or directory")
    cluster_parser.add_argument("input", type=Path, help="Path to a .smi, .smi.gz, .npy file, or a directory")
    cluster_parser.add_argument("--threshold", type=float, default=None)
    cluster_parser.add_argument("--fp-type", default=_config.FINGERPRINT_TYPE)
    cluster_parser.add_argument("--n-bits", type=int, default=_config.N_BITS)
    cluster_parser.add_argument("--branching-factor", type=int, default=_config.BRANCHING_FACTOR)
    cluster_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION)
    cluster_parser.add_argument("--recluster-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_INITIAL)
    cluster_parser.add_argument("--recluster-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD)
    cluster_parser.add_argument("--force-sequential", action=argparse.BooleanOptionalAction, default=False)
    cluster_parser.add_argument("--save-tree", action=argparse.BooleanOptionalAction, default=_config.SAVE_TREE)
    cluster_parser.add_argument("--save-centroids", action=argparse.BooleanOptionalAction, default=_config.SAVE_CENTROIDS)
    cluster_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    multiround_parser = subparsers.add_parser("multiround", help="Run multi-round clustering")
    multiround_parser.add_argument("input", nargs="+", type=Path, help="Input .npy files or a directory containing them")
    multiround_parser.add_argument("--out-dir", type=Path, required=True)
    multiround_parser.add_argument("--n-features", type=int, default=None)
    multiround_parser.add_argument("--input-is-packed", action=argparse.BooleanOptionalAction, default=True)
    multiround_parser.add_argument("--num-initial-processes", type=int, default=10)
    multiround_parser.add_argument("--num-midsection-processes", type=int, default=None)
    multiround_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION)
    multiround_parser.add_argument("--branching-factor", type=int, default=_config.BRANCHING_FACTOR)
    multiround_parser.add_argument("--threshold", type=float, default=_config.THRESHOLD)
    multiround_parser.add_argument("--midsection-threshold-change", type=float, default=_config.MIDSECTION_THRESHOLD_CHANGE)
    multiround_parser.add_argument("--num-midsection-rounds", type=int, default=_config.NUM_MIDSECTION_ROUNDS)
    multiround_parser.add_argument("--bin-size", type=int, default=_config.BIN_SIZE)
    multiround_parser.add_argument("--max-tasks-per-process", type=int, default=1)
    multiround_parser.add_argument("--save-tree", action=argparse.BooleanOptionalAction, default=_config.SAVE_TREE)
    multiround_parser.add_argument("--save-centroids", action=argparse.BooleanOptionalAction, default=_config.SAVE_CENTROIDS)
    multiround_parser.add_argument("--reclustering-iterations-initial", type=int, default=_config.RECLUSTERING_ITERATIONS_INITIAL)
    multiround_parser.add_argument("--reclustering-iterations-midsection", type=int, default=_config.RECLUSTERING_ITERATIONS_MIDSECTION)
    multiround_parser.add_argument("--reclustering-iterations-final", type=int, default=_config.RECLUSTERING_ITERATIONS_FINAL)
    multiround_parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD)
    multiround_parser.add_argument("--max-fps", type=int, default=None)
    multiround_parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    multiround_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    initial_round_parser = subparsers.add_parser(
        "initial-round", help="Run initial round HPC clustering with automatic job submission"
    )
    initial_round_parser.add_argument("input", nargs="+", type=Path, help="Input .smi or .smi.gz files")
    initial_round_parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for job scripts and logs")
    initial_round_parser.add_argument("--files-per-job", type=int, default=_config.FILES_PER_JOB, help="Number of files per job")
    initial_round_parser.add_argument("--max-jobs-per-script", type=int, default=_config.MAX_JOBS_PER_SCRIPT, help="Maximum jobs per submission script")
    initial_round_parser.add_argument("--threshold", type=float, default=_config.THRESHOLD, help="BitBirch threshold")
    initial_round_parser.add_argument("--branching-factor", type=int, default=_config.BRANCHING_FACTOR, help="BitBirch branching factor")
    initial_round_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION, help="Merge criterion")
    initial_round_parser.add_argument("--fp-type", default=_config.FINGERPRINT_TYPE, help="Fingerprint type")
    initial_round_parser.add_argument("--n-bits", type=int, default=_config.N_BITS, help="Number of fingerprint bits")
    initial_round_parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_INITIAL, help="Reclustering iterations")
    initial_round_parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold for reclustering")
    initial_round_parser.add_argument("--slurm-mem", default=_config.SLURM_MEM_INITIAL, help="SLURM memory allocation")
    initial_round_parser.add_argument("--slurm-cpus", type=int, default=_config.SLURM_CPUS_INITIAL, help="SLURM CPU count")
    initial_round_parser.add_argument("--slurm-time", default=_config.SLURM_TIME, help="SLURM time limit")
    initial_round_parser.add_argument("--slurm-partition", default=_config.SLURM_PARTITION, help="SLURM partition (optional)")
    initial_round_parser.add_argument("--result-base-dir", type=Path, default=None, help="Base directory for results (optional)")
    initial_round_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    initial_round_parser.add_argument(
        "--code-ids",
        action="store_true",
        default=False,
        help="Read SMILES and ZINC_ID records (comma, tab, or space separated) as packed molecule indices",
    )

    midsection_round_parser = subparsers.add_parser(
        "midsection-round", help="Run midsection round HPC clustering"
    )
    midsection_round_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory containing previous round results")
    midsection_round_parser.add_argument("--round-idx", type=int, required=True, help="Current round index")
    midsection_round_parser.add_argument("--bin-size", type=int, default=_config.BIN_SIZE, help="Number of buffer/index pairs per batch job")
    midsection_round_parser.add_argument("--max-jobs-per-script", type=int, default=_config.MAX_JOBS_PER_SCRIPT, help="Maximum jobs per submission script")
    midsection_round_parser.add_argument("--threshold", type=float, default=None, help="BitBirch threshold (default: same as config)")
    midsection_round_parser.add_argument("--branching-factor", type=int, default=None, help="BitBirch branching factor (default: same as config)")
    midsection_round_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION, help="Merge criterion")
    midsection_round_parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_MIDSECTION, help="Reclustering iterations")
    midsection_round_parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold for reclustering")
    midsection_round_parser.add_argument("--slurm-mem", default=_config.SLURM_MEM_MIDSECTION, help="SLURM memory allocation")
    midsection_round_parser.add_argument("--slurm-cpus", type=int, default=_config.SLURM_CPUS_MIDSECTION, help="SLURM CPU count")
    midsection_round_parser.add_argument("--slurm-time", default=_config.SLURM_TIME, help="SLURM time limit")
    midsection_round_parser.add_argument("--slurm-partition", default=_config.SLURM_PARTITION, help="SLURM partition (optional)")
    midsection_round_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    final_round_parser = subparsers.add_parser(
        "final-round", help="Run final round HPC clustering"
    )
    final_round_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory containing previous round results")
    final_round_parser.add_argument("--prev-round-idx", type=int, required=True, help="Previous round index to read from")
    final_round_parser.add_argument("--threshold", type=float, default=None, help="BitBirch threshold (default: same as config)")
    final_round_parser.add_argument("--branching-factor", type=int, default=None, help="BitBirch branching factor (default: same as config)")
    final_round_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION, help="Merge criterion")
    final_round_parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_FINAL, help="Reclustering iterations")
    final_round_parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold for reclustering")
    final_round_parser.add_argument("--save-npy", action=argparse.BooleanOptionalAction, default=False, help="Save clusters as separate .npy files (one per cluster)")
    final_round_parser.add_argument("--save-tree", action=argparse.BooleanOptionalAction, default=False, help="Save the BitBirch tree")
    final_round_parser.add_argument("--save-centroids", action=argparse.BooleanOptionalAction, default=False, help="Save centroids and cluster assignments")
    final_round_parser.add_argument("--slurm-mem", default=_config.SLURM_MEM_FINAL, help="SLURM memory allocation")
    final_round_parser.add_argument("--slurm-cpus", type=int, default=_config.SLURM_CPUS_FINAL, help="SLURM CPU count")
    final_round_parser.add_argument("--slurm-time", default=_config.SLURM_TIME, help="SLURM time limit")
    final_round_parser.add_argument("--slurm-partition", default=_config.SLURM_PARTITION, help="SLURM partition (optional)")
    final_round_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    # Fingerprint generation commands
    binary_fps_parser = subparsers.add_parser(
        "binary-fps", help="Generate binary fingerprints from SMILES"
    )
    binary_fps_parser.add_argument(
        "input", type=Path, help="Input .smi, .smi.gz file or directory containing them"
    )
    binary_fps_parser.add_argument(
        "--fp-type", default="ECFP4",
        help="Fingerprint type ['RDKIT', 'ECFP4', 'ECFP6', 'AP', 'TT', 'MACCS']"
    )
    binary_fps_parser.add_argument(
        "--n-bits", type=int, default=2048,
        help="Number of bits for fingerprint (ignored for MACCS)"
    )
    binary_fps_parser.add_argument(
        "--packed", action=argparse.BooleanOptionalAction, default=True,
        help="Return packed fingerprints (default: True)"
    )
    binary_fps_parser.add_argument(
        "--return-invalid", action=argparse.BooleanOptionalAction, default=True,
        help="Return indices of invalid SMILES"
    )
    binary_fps_parser.add_argument(
        "--standarize", action=argparse.BooleanOptionalAction, default=False,
        help="Standardize molecules before generating fingerprints"
    )
    binary_fps_parser.add_argument(
        "--n-processes", type=int, default=None,
        help="Number of processes for parallelization (default: min(available_cpus, 32))"
    )
    binary_fps_parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .npy file for fingerprints (default: binary_fps.npy)"
    )

    count_fps_parser = subparsers.add_parser(
        "count-fps", help="Generate count fingerprints from SMILES"
    )
    count_fps_parser.add_argument(
        "input", type=Path, help="Input .smi, .smi.gz file or directory containing them"
    )
    count_fps_parser.add_argument(
        "--fp-type", default="ECFP4",
        help="Fingerprint type ['RDKIT', 'ECFP4', 'ECFP6']"
    )
    count_fps_parser.add_argument(
        "--n-bits", type=int, default=2048,
        help="Number of bits for fingerprint"
    )
    count_fps_parser.add_argument(
        "--return-invalid", action=argparse.BooleanOptionalAction, default=True,
        help="Return indices of invalid SMILES"
    )
    count_fps_parser.add_argument(
        "--n-processes", type=int, default=None,
        help="Number of processes for parallelization (default: min(available_cpus, 32))"
    )
    count_fps_parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .npy file for fingerprints (default: count_fps.npy)"
    )

    real_fps_parser = subparsers.add_parser(
        "real-fps", help="Generate real-valued fingerprints from RDKit descriptors"
    )
    real_fps_parser.add_argument(
        "input", type=Path, help="Input .smi, .smi.gz file or directory containing them"
    )
    real_fps_parser.add_argument(
        "--return-invalid", action=argparse.BooleanOptionalAction, default=True,
        help="Return indices of invalid SMILES"
    )
    real_fps_parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .npy file for fingerprints (default: real_fps.npy)"
    )

    cluster_sampling_parser = subparsers.add_parser(
        "cluster-sampling", help="Sample molecules from clusters"
    )
    cluster_sampling_parser.add_argument(
        "--clusters", type=Path, required=True,
        help="Path to .pkl file containing cluster assignments"
    )
    cluster_sampling_parser.add_argument(
        "--fingerprints", type=Path, default=None,
        help="Path to .npy file or directory containing fingerprints"
    )
    cluster_sampling_parser.add_argument(
        "--smiles", type=Path, default=None,
        help="Path to .smi or .smi.gz file or directory containing SMILES files"
    )
    cluster_sampling_parser.add_argument(
        "--centroids", type=Path, default=None,
        help="Path to .pkl file containing centroids (required for centroid-like sampling)"
    )
    cluster_sampling_parser.add_argument(
        "--method", default="centroid-like",
        choices=["singletons", "medoids", "centroid-like"],
        help="Sampling method (default: centroid-like)"
    )
    cluster_sampling_parser.add_argument(
        "--min-size", type=int, default=0,
        help="Minimum cluster size to sample from (for medoids method)"
    )
    cluster_sampling_parser.add_argument(
        "--fp-type", default="ECFP4",
        help="Fingerprint type for computing from SMILES (when fps not provided)"
    )
    cluster_sampling_parser.add_argument(
        "--n-bits", type=int, default=2048,
        help="Number of bits in fingerprint vectors"
    )
    cluster_sampling_parser.add_argument(
        "--sample", action=argparse.BooleanOptionalAction, default=True,
        help="Subsample large clusters before computing medoid/centroid"
    )
    cluster_sampling_parser.add_argument(
        "--sample-min-size", type=int, default=1000,
        help="Subsample threshold for large clusters"
    )
    cluster_sampling_parser.add_argument(
        "--n-processes", type=int, default=None,
        help="Number of processes for parallel sampling (default: min(8, cpu_count())). Only applies when fps are provided."
    )

    rewrite_smiles_parser = subparsers.add_parser(
        "rewrite-smiles-by-cluster", help="Reorganize SMILES files by cluster"
    )
    rewrite_smiles_parser.add_argument(
        "--clusters", type=Path, required=True,
        help="Path to .pkl file containing cluster assignments (list of lists)"
    )
    rewrite_smiles_parser.add_argument(
        "--smiles-dir", type=Path, required=True,
        help="Directory containing input SMILES files (*.smi or *.smi.gz)"
    )
    rewrite_smiles_parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write cluster-organized SMILES files"
    )
    rewrite_smiles_parser.add_argument(
        "--smiles-per-file", type=int, default=1_000_000,
        help="Number of SMILES per input file (default: 1M)"
    )
    rewrite_smiles_parser.add_argument(
        "--compressed", action=argparse.BooleanOptionalAction, default=False,
        help="Write gzipped files (.smi.gz) instead of plain text"
    )
    rewrite_smiles_parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Number of parallel processes (default: 8)"
    )
    rewrite_smiles_parser.add_argument(
        "--start-at", type=int, default=0,
        help="Cluster index to start processing from (default: 0)"
    )

    rewrite_smiles_npy_parser = subparsers.add_parser(
        "rewrite-smiles-by-cluster-npy", help="Reorganize SMILES files by cluster using cluster_<id>.npy files"
    )
    rewrite_smiles_npy_parser.add_argument(
        "--clusters-dir", type=Path, required=True,
        help="Directory containing cluster_<id>.npy files"
    )
    rewrite_smiles_npy_parser.add_argument(
        "--smiles-dir", type=Path, required=True,
        help="Directory containing input SMILES files (*.smi or *.smi.gz)"
    )
    rewrite_smiles_npy_parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write cluster-organized SMILES files"
    )
    rewrite_smiles_npy_parser.add_argument(
        "--smiles-per-file", type=int, default=1_000_000,
        help="Number of SMILES per input file (default: 1M)"
    )
    rewrite_smiles_npy_parser.add_argument(
        "--compressed", action=argparse.BooleanOptionalAction, default=False,
        help="Write gzipped files (.smi.gz) instead of plain text"
    )
    rewrite_smiles_npy_parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Number of parallel processes (default: 8)"
    )
    rewrite_smiles_npy_parser.add_argument(
        "--start-at", type=int, default=0,
        help="Cluster index to start processing from (default: 0)"
    )
    rewrite_smiles_npy_parser.add_argument(
        "--write-database-ids", action=argparse.BooleanOptionalAction, default=False,
        help="Write ordered database IDs alongside cluster SMILES outputs"
    )

    rewrite_smiles_single_npy_parser = subparsers.add_parser(
        "rewrite-smiles-single-cluster-npy",
        help="Reorganize SMILES for one cluster_<id>.npy file"
    )
    rewrite_smiles_single_npy_parser.add_argument(
        "--cluster-file", type=Path, required=True,
        help="Path to one cluster_<id>.npy file"
    )
    rewrite_smiles_single_npy_parser.add_argument(
        "--smiles-dir", type=Path, required=True,
        help="Directory containing input SMILES files (*.smi or *.smi.gz)"
    )
    rewrite_smiles_single_npy_parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write cluster-organized SMILES files"
    )
    rewrite_smiles_single_npy_parser.add_argument(
        "--smiles-per-file", type=int, default=1_000_000,
        help="Number of SMILES per input file (default: 1M)"
    )
    rewrite_smiles_single_npy_parser.add_argument(
        "--compressed", action=argparse.BooleanOptionalAction, default=False,
        help="Write gzipped files (.smi.gz) instead of plain text"
    )
    rewrite_smiles_single_npy_parser.add_argument(
        "--write-database-ids", action=argparse.BooleanOptionalAction, default=False,
        help="Write ordered database IDs alongside cluster SMILES outputs"
    )
    rewrite_smiles_single_npy_parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False,
        help="Overwrite existing cluster output files"
    )

    rewrite_smiles_submit_parser = subparsers.add_parser(
        "rewrite-smiles-by-cluster-npy-submit",
        help="Generate SLURM submission script to rewrite one job per cluster_<id>.npy"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--clusters-dir", type=Path, required=True,
        help="Directory containing cluster_<id>.npy files"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--smiles-dir", type=Path, required=True,
        help="Directory containing input SMILES files (*.smi or *.smi.gz)"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write cluster-organized SMILES files"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--submit-dir", type=Path, default=None,
        help="Directory to write submission script and logs (default: output-dir)"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--smiles-per-file", type=int, default=1_000_000,
        help="Number of SMILES per input file (default: 1M)"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--compressed", action=argparse.BooleanOptionalAction, default=True,
        help="Write gzipped files (.smi.gz)"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--write-database-ids", action=argparse.BooleanOptionalAction, default=False,
        help="Write ordered database IDs alongside cluster SMILES outputs"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False,
        help="Overwrite existing cluster output files"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--max-jobs-per-script", type=int, default=_config.MAX_JOBS_PER_SCRIPT,
        help="Maximum number of jobs per submission script"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--slurm-mem", default=_config.SLURM_MEM_INITIAL,
        help="SLURM memory allocation"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--slurm-cpus", type=int, default=_config.SLURM_CPUS_INITIAL,
        help="SLURM CPU count"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--slurm-time", default=_config.SLURM_TIME,
        help="SLURM time limit"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--slurm-partition", default=_config.SLURM_PARTITION,
        help="SLURM partition (optional)"
    )
    rewrite_smiles_submit_parser.add_argument(
        "--conda-env", default="iChem",
        help="Conda environment to activate in each submitted job"
    )

    sample_from_cluster_files_parser = subparsers.add_parser(
        "sample-from-cluster-files", help="Sample molecules from cluster files directory"
    )
    sample_from_cluster_files_parser.add_argument(
        "--cluster-dir", type=Path, required=True,
        help="Directory containing cluster files (.smi or .smi.gz)"
    )
    sample_from_cluster_files_parser.add_argument(
        "--method", default="medoids",
        choices=["medoids", "centroid-like"],
        help="Sampling method (default: medoids)"
    )
    sample_from_cluster_files_parser.add_argument(
        "--centroids", type=Path, default=None,
        help="Path to .pkl file containing centroids (required for centroid-like method)"
    )
    sample_from_cluster_files_parser.add_argument(
        "--fp-type", default="ECFP4",
        help="Fingerprint type (default: ECFP4)"
    )
    sample_from_cluster_files_parser.add_argument(
        "--n-bits", type=int, default=2048,
        help="Number of fingerprint bits (default: 2048)"
    )
    sample_from_cluster_files_parser.add_argument(
        "--sample", action=argparse.BooleanOptionalAction, default=True,
        help="Subsample large clusters before computing medoid/centroid (default: True)"
    )
    sample_from_cluster_files_parser.add_argument(
        "--sample-min-size", type=int, default=1000,
        help="Subsample threshold for large clusters (default: 1000)"
    )
    sample_from_cluster_files_parser.add_argument(
        "--n-processes", type=int, default=None,
        help="Number of parallel processes (default: min(8, cpu_count()))"
    )
    sample_from_cluster_files_parser.add_argument(
        "--output", type=Path, default=None,
        help="Output file path for sampled SMILES (optional)"
    )

    return parser


def _run_cluster(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    print(f"[cluster] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    result = cluster(
        args.input,
        threshold=args.threshold,
        fp_type=args.fp_type,
        n_bits=args.n_bits,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        recluster_iterations=args.recluster_iterations,
        recluster_extra_threshold=args.recluster_extra_threshold,
        verbose=args.verbose,
        force_sequential=args.force_sequential,
        save_tree=args.save_tree,
        save_centroids=args.save_centroids,
    )

    elapsed = time.perf_counter() - t0
    print(f"[cluster] Finished in {elapsed:.2f} s")

    # 0 means multiround path; non-zero sequential results are saved in cluster.py
    if result == 0:
        print("Multiround clustering completed. Results saved by multiround handler.")
    else:
        print("Sequential clustering completed. Results saved as clusters.pkl near the input fingerprints.")
    return 0


def _run_multiround(args: argparse.Namespace) -> int:
    input_files: list[Path] = []
    for value in args.input:
        input_files.extend(_path_list(str(value)))
    if not input_files:
        raise ValueError("No input files were found")

    t0 = time.perf_counter()
    print(f"[multiround] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    timer = run_multiround_reclustering(
        input_files=input_files,
        out_dir=args.out_dir,
        n_features=args.n_features,
        input_is_packed=args.input_is_packed,
        num_initial_processes=args.num_initial_processes,
        num_midsection_processes=args.num_midsection_processes,
        merge_criterion=args.merge_criterion,
        branching_factor=args.branching_factor,
        threshold=args.threshold,
        midsection_threshold_change=args.midsection_threshold_change,
        num_midsection_rounds=args.num_midsection_rounds,
        bin_size=args.bin_size,
        max_tasks_per_process=args.max_tasks_per_process,
        save_tree=args.save_tree,
        save_centroids=args.save_centroids,
        reclustering_iterations_initial=args.reclustering_iterations_initial,
        reclustering_iterations_midsection=args.reclustering_iterations_midsection,
        reclustering_iterations_final=args.reclustering_iterations_final,
        reclustering_extra_threshold=args.reclustering_extra_threshold,
        max_fps=args.max_fps,
        verbose=args.verbose,
        cleanup=args.cleanup,
    )
    elapsed = time.perf_counter() - t0
    print(f"[multiround] Finished in {elapsed:.2f} s")
    print(timer)
    return 0


def _run_initial_round(args: argparse.Namespace) -> int:
    input_files = [Path(f) for f in args.input]
    if not input_files:
        raise ValueError("No input files were found")

    script_paths = prepare_initial_round_jobs(
        input_files=input_files,
        output_dir=args.out_dir,
        files_per_job=args.files_per_job,
        threshold=args.threshold,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        fp_type=args.fp_type,
        n_bits=args.n_bits,
        reclustering_iterations=args.reclustering_iterations,
        extra_threshold=args.reclustering_extra_threshold,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        result_base_dir=args.result_base_dir,
        max_jobs_per_script=args.max_jobs_per_script,
        verbose=args.verbose,
        code_ids=args.code_ids,
    )

    # Handle single or multiple scripts
    if isinstance(script_paths, list):
        print(f"\n✓ Generated {len(script_paths)} submission scripts:")
        for i, path in enumerate(script_paths, 1):
            print(f"  {i}. bash {Path(path).resolve()}")
        print(f"\nRun each script to submit batches of jobs (max {args.max_jobs_per_script} per script)")
    else:
        print(f"\n✓ Generated submission script: {script_paths}")
        print(f"Run the following to submit all initial round jobs:")
        print(f"\n  bash {Path(script_paths).resolve()}\n")
    return 0


def _run_midsection_round(args: argparse.Namespace) -> int:
    script_paths = prepare_midsection_round_jobs(
        output_dir=args.output_dir,
        round_idx=args.round_idx,
        bin_size=args.bin_size,
        threshold=args.threshold,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        reclustering_iterations=args.reclustering_iterations,
        reclustering_extra_threshold=args.reclustering_extra_threshold,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        max_jobs_per_script=args.max_jobs_per_script,
        verbose=args.verbose,
    )

    # Handle single or multiple scripts
    if isinstance(script_paths, list):
        print(f"\n✓ Generated {len(script_paths)} submission scripts:")
        for i, path in enumerate(script_paths, 1):
            print(f"  {i}. bash {Path(path).resolve()}")
        print(f"\nRun each script to submit batches of jobs (max {args.max_jobs_per_script} per script)")
    else:
        print(f"\n✓ Generated submission script: {script_paths}")
        print(f"Run the following to submit all midsection round jobs:")
        print(f"\n  bash {Path(script_paths).resolve()}\n")
    return 0


def _run_final_round(args: argparse.Namespace) -> int:
    script_path = prepare_final_round_job(
        output_dir=args.output_dir,
        prev_round_idx=args.prev_round_idx,
        threshold=args.threshold,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        reclustering_iterations=args.reclustering_iterations,
        reclustering_extra_threshold=args.reclustering_extra_threshold,
        save_npy=args.save_npy,
        save_tree=args.save_tree,
        save_centroids=args.save_centroids,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        verbose=args.verbose,
    )

    print(f"\n✓ Generated submission script: {script_path}")
    print(f"Run the following to submit the final round job:")
    print(f"\n  bash {Path(script_path).resolve()}\n")
    return 0


def _run_binary_fps(args: argparse.Namespace) -> int:
    smi_files = get_smi_files(args.input)
    
    if not smi_files:
        raise ValueError(f"No .smi or .smi.gz files found in {args.input}")
    
    output_dir = args.input if args.input.is_dir() else args.input.parent
    
    for smi_file in smi_files:
        print(f"\nProcessing: {smi_file.name}")
        print("Loading SMILES...")
        smiles = load_smiles(smi_file)
        print(f"Loaded {len(smiles)} SMILES")

        print("Generating binary fingerprints...")
        t0 = time.perf_counter()
        if args.return_invalid:
            fps, invalid_indices = binary_fps(
                smiles,
                fp_type=args.fp_type,
                n_bits=args.n_bits,
                return_invalid=True,
                standarize=args.standarize,
                packed=args.packed,
                n_processes=args.n_processes,
            )
            if invalid_indices:
                print(f"Warning: {len(invalid_indices)} invalid SMILES found")
                invalid_path = output_dir / f"{smi_file.stem}_invalid_indices.json"
                with open(invalid_path, "w") as f:
                    json.dump(invalid_indices, f, indent=2)
                print(f"✓ Saved {len(invalid_indices)} invalid indices to {invalid_path}")
        else:
            fps = binary_fps(
                smiles,
                fp_type=args.fp_type,
                n_bits=args.n_bits,
                return_invalid=False,
                standarize=args.standarize,
                packed=args.packed,
                n_processes=args.n_processes,
            )
        elapsed = time.perf_counter() - t0
        print(f"✓ Generated in {elapsed:.2f}s")

        if args.out:
            output_path = args.out
        else:
            output_path = get_output_path(smi_file, args.fp_type, '.npy', output_dir)
        
        np.save(output_path, fps)
        print(f"✓ Saved {fps.shape[0]} fingerprints to {output_path}")

    return 0


def _run_count_fps(args: argparse.Namespace) -> int:
    smi_files = get_smi_files(args.input)
    
    if not smi_files:
        raise ValueError(f"No .smi or .smi.gz files found in {args.input}")
    
    output_dir = args.input if args.input.is_dir() else args.input.parent
    
    for smi_file in smi_files:
        print(f"\nProcessing: {smi_file.name}")
        print("Loading SMILES...")
        smiles = load_smiles(smi_file)
        print(f"Loaded {len(smiles)} SMILES")

        print("Generating count fingerprints...")
        t0 = time.perf_counter()
        if args.return_invalid:
            fps, invalid_indices = count_fps(
                smiles,
                fp_type=args.fp_type,
                n_bits=args.n_bits,
                return_invalid=True,
                n_processes=args.n_processes,
            )
            if invalid_indices:
                print(f"Warning: {len(invalid_indices)} invalid SMILES found")
                invalid_path = output_dir / f"{smi_file.stem}_invalid_indices.json"
                with open(invalid_path, "w") as f:
                    json.dump(invalid_indices, f, indent=2)
                print(f"✓ Saved {len(invalid_indices)} invalid indices to {invalid_path}")
        else:
            fps = count_fps(
                smiles,
                fp_type=args.fp_type,
                n_bits=args.n_bits,
                return_invalid=False,
                n_processes=args.n_processes,
            )
        elapsed = time.perf_counter() - t0
        print(f"✓ Generated in {elapsed:.2f}s")

        if args.out:
            output_path = args.out
        else:
            output_path = get_output_path(smi_file, args.fp_type, '.npy', output_dir)
        
        np.save(output_path, fps)
        print(f"✓ Saved {fps.shape[0]} fingerprints to {output_path}")

    return 0


def _run_real_fps(args: argparse.Namespace) -> int:
    smi_files = get_smi_files(args.input)

    if not smi_files:
        raise ValueError(f"No .smi or .smi.gz files found in {args.input}")

    output_dir = args.input if args.input.is_dir() else args.input.parent

    for smi_file in smi_files:
        print(f"\nProcessing: {smi_file.name}")
        print("Loading SMILES...")
        smiles = load_smiles(smi_file)
        print(f"Loaded {len(smiles)} SMILES")

        print("Generating real-valued fingerprints...")
        t0 = time.perf_counter()
        if args.return_invalid:
            fps, invalid_indices = real_fps(
                smiles, return_invalid=True
            )
            if invalid_indices:
                print(f"Warning: {len(invalid_indices)} invalid SMILES found")
                invalid_path = output_dir / f"{smi_file.stem}_invalid_indices.json"
                with open(invalid_path, "w") as f:
                    json.dump(invalid_indices, f, indent=2)
                print(f"✓ Saved {len(invalid_indices)} invalid indices to {invalid_path}")
        else:
            fps = real_fps(smiles, return_invalid=False)
        elapsed = time.perf_counter() - t0
        print(f"✓ Generated in {elapsed:.2f}s")

        if args.out:
            output_path = args.out
        else:
            output_path = get_output_path(smi_file, 'RDKitDescriptors', '.npy', output_dir)

        np.save(output_path, fps)
        print(f"✓ Saved {fps.shape[0]} fingerprints to {output_path}")

    return 0


def _run_cluster_sampling(args: argparse.Namespace) -> int:
    print(f"[cluster-sampling] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.perf_counter()

    print(f"Sampling method: {args.method}")
    print(f"Loading clusters from {args.clusters}...")

    sampled_data = sample_clusters(
        clusters=args.clusters,
        sampling_method=args.method,
        fps=args.fingerprints,
        smiles=args.smiles,
        centroids=args.centroids,
        min_size=args.min_size,
        fp_type=args.fp_type,
        n_bits=args.n_bits,
        sample=args.sample,
        sample_min_size=args.sample_min_size,
        n_processes=args.n_processes,
    )

    elapsed = time.perf_counter() - t0
    print(f"✓ Sampling completed in {elapsed:.2f}s")
    print(f"✓ Sampled {len(sampled_data)} molecules")

    if args.smiles and isinstance(sampled_data[0], str):
        smi_path = Path(args.smiles)
        stem = smi_path.stem
        if stem.endswith('.smi'):
            stem = stem[:-4]

        output_dir = smi_path.parent if smi_path.is_file() else smi_path
        output_path = output_dir / f"{stem}_{args.method}.smi"

        with open(output_path, 'w') as f:
            for smi in sampled_data:
                f.write(f"{smi}\n")

        print(f"✓ Saved sampled SMILES to {output_path}")

    print(f"[cluster-sampling] Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


def _run_rewrite_smiles_by_cluster(args: argparse.Namespace) -> int:
    print(f"[rewrite-smiles-by-cluster] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    t0 = time.perf_counter()

    print(f"Loading clusters from {args.clusters}...")
    sys.stdout.flush()
    with open(args.clusters, 'rb') as f:
        clusters = pkl.load(f)
    print(f"Loaded {len(clusters)} clusters")
    sys.stdout.flush()

    # Auto-detect starting point if output dir exists
    start_at = args.start_at
    if args.output_dir.exists() and start_at == 0:
        first_missing = find_first_missing_cluster(
            str(args.output_dir), len(clusters), args.compressed
        )
        if first_missing < len(clusters):
            print(f"Output directory exists. Found {first_missing} completed clusters.")
            print(f"Resuming from cluster {first_missing}")
            sys.stdout.flush()
            start_at = first_missing

    print(f"Reorganizing SMILES files by cluster...")
    sys.stdout.flush()
    rewrite_smiles_by_cluster(
        clusters=clusters,
        input_smiles_dir=str(args.smiles_dir),
        output_dir=str(args.output_dir),
        smiles_per_file=args.smiles_per_file,
        compressed=args.compressed,
        num_workers=args.num_workers,
        start_at=start_at,
    )

    elapsed = time.perf_counter() - t0
    print(f"✓ Reorganization completed in {elapsed:.2f}s")
    sys.stdout.flush()
    print(f"[rewrite-smiles-by-cluster] Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    return 0


def _run_rewrite_smiles_by_cluster_npy(args: argparse.Namespace) -> int:
    print(f"[rewrite-smiles-by-cluster-npy] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    t0 = time.perf_counter()

    if not args.clusters_dir.exists():
        print(f"✗ Error: Cluster directory not found: {args.clusters_dir}")
        return 1

    cluster_files = sorted(args.clusters_dir.glob("cluster_*.npy"), key=lambda path: int(path.stem.split("_")[-1]))
    if not cluster_files:
        print(f"✗ Error: No cluster_*.npy files found in {args.clusters_dir}")
        return 1

    print(f"Found {len(cluster_files)} cluster files")
    sys.stdout.flush()

    start_at = args.start_at
    if args.output_dir.exists() and start_at == 0:
        first_missing = find_first_missing_cluster_from_files(
            str(args.output_dir), cluster_files, args.compressed
        )
        last_cluster_id = int(cluster_files[-1].stem.split("_")[-1])
        if first_missing <= last_cluster_id:
            print(f"Output directory exists. Found clusters up to {first_missing - 1} completed.")
            print(f"Resuming from cluster {first_missing}")
            sys.stdout.flush()
            start_at = first_missing

    print(f"Reorganizing SMILES files by cluster from {start_at}...")
    sys.stdout.flush()
    rewrite_smiles_by_cluster_from_npy_dir(
        clusters_dir=str(args.clusters_dir),
        input_smiles_dir=str(args.smiles_dir),
        output_dir=str(args.output_dir),
        smiles_per_file=args.smiles_per_file,
        compressed=args.compressed,
        num_workers=args.num_workers,
        start_at=start_at,
        write_database_ids=args.write_database_ids,
    )

    elapsed = time.perf_counter() - t0
    print(f"✓ Reorganization completed in {elapsed:.2f}s")
    sys.stdout.flush()
    print(f"[rewrite-smiles-by-cluster-npy] Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    return 0


def _run_rewrite_smiles_single_cluster_npy(args: argparse.Namespace) -> int:
    print(f"[rewrite-smiles-single-cluster-npy] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    t0 = time.perf_counter()

    rewrite_single_cluster_from_npy(
        cluster_file=str(args.cluster_file),
        input_smiles_dir=str(args.smiles_dir),
        output_dir=str(args.output_dir),
        smiles_per_file=args.smiles_per_file,
        compressed=args.compressed,
        write_database_ids=args.write_database_ids,
        overwrite=args.overwrite,
    )

    elapsed = time.perf_counter() - t0
    print(f"✓ Cluster rewrite completed in {elapsed:.2f}s")
    sys.stdout.flush()
    print(f"[rewrite-smiles-single-cluster-npy] Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    return 0


def _run_rewrite_smiles_by_cluster_npy_submit(args: argparse.Namespace) -> int:
    script_path = prepare_rewrite_smiles_npy_jobs(
        clusters_dir=str(args.clusters_dir),
        input_smiles_dir=str(args.smiles_dir),
        output_dir=str(args.output_dir),
        submit_dir=str(args.submit_dir) if args.submit_dir else None,
        smiles_per_file=args.smiles_per_file,
        compressed=args.compressed,
        write_database_ids=args.write_database_ids,
        overwrite=args.overwrite,
        max_jobs_per_script=args.max_jobs_per_script,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        conda_env=args.conda_env,
    )

    if isinstance(script_path, list):
        print(f"\n✓ Generated {len(script_path)} rewrite submission scripts:")
        for i, path in enumerate(script_path, 1):
            print(f"  {i}. bash {Path(path).resolve()}")
        print(f"\nRun each script to submit jobs in batches of up to {args.max_jobs_per_script} per script")
    else:
        print(f"\n✓ Generated rewrite submission script: {script_path}")
        print("Run the following to submit one job per cluster file:")
        print(f"\n  bash {Path(script_path).resolve()}\n")
    return 0


def _run_sample_from_cluster_files(args: argparse.Namespace) -> int:
    print(f"[sample-from-cluster-files] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    t0 = time.perf_counter()

    # Validate cluster directory
    if not args.cluster_dir.is_dir():
        print(f"✗ Error: Cluster directory not found: {args.cluster_dir}")
        return 1

    print(f"Cluster directory: {args.cluster_dir}")
    print(f"Sampling method: {args.method}")
    sys.stdout.flush()

    # Load centroids if needed
    centroids = None
    if args.method == "centroid-like":
        if args.centroids is None:
            print("✗ Error: --centroids is required for centroid-like sampling")
            return 1
        if not args.centroids.exists():
            print(f"✗ Error: Centroids file not found: {args.centroids}")
            return 1
        print(f"Loading centroids from {args.centroids}...")
        with open(args.centroids, 'rb') as f:
            centroids = pkl.load(f)
        print(f"Loaded {len(centroids)} centroids")
        sys.stdout.flush()

    # Perform sampling
    print("Sampling molecules from cluster files...")
    sys.stdout.flush()
    sampled_smiles = sample_from_cluster_files(
        cluster_dir=args.cluster_dir,
        sampling_method=args.method,
        centroids=centroids,
        fp_type=args.fp_type,
        n_bits=args.n_bits,
        sample=args.sample,
        sample_min_size=args.sample_min_size,
        n_processes=args.n_processes,
    )

    elapsed = time.perf_counter() - t0
    print(f"✓ Sampling completed in {elapsed:.2f}s")
    print(f"✓ Sampled {len(sampled_smiles)} molecules (one per cluster)")
    sys.stdout.flush()

    # Save output if requested
    if args.output:
        print(f"Saving sampled SMILES to {args.output}...")
        with open(args.output, 'w') as f:
            for smi in sampled_smiles:
                f.write(f"{smi}\n")
        print(f"✓ Saved {len(sampled_smiles)} SMILES to {args.output}")
    else:
        # Print to stdout if no output file specified
        for smi in sampled_smiles:
            print(smi)

    sys.stdout.flush()
    print(f"[sample-from-cluster-files] Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _print_banner()

    if args.command == "cluster":
        return _run_cluster(args)
    if args.command == "multiround":
        return _run_multiround(args)
    if args.command == "initial-round":
        return _run_initial_round(args)
    if args.command == "midsection-round":
        return _run_midsection_round(args)
    if args.command == "final-round":
        return _run_final_round(args)
    if args.command == "binary-fps":
        return _run_binary_fps(args)
    if args.command == "count-fps":
        return _run_count_fps(args)
    if args.command == "real-fps":
        return _run_real_fps(args)
    if args.command == "cluster-sampling":
        return _run_cluster_sampling(args)
    if args.command == "rewrite-smiles-by-cluster":
        return _run_rewrite_smiles_by_cluster(args)
    if args.command == "rewrite-smiles-by-cluster-npy":
        return _run_rewrite_smiles_by_cluster_npy(args)
    if args.command == "rewrite-smiles-single-cluster-npy":
        return _run_rewrite_smiles_single_cluster_npy(args)
    if args.command == "rewrite-smiles-by-cluster-npy-submit":
        return _run_rewrite_smiles_by_cluster_npy_submit(args)
    if args.command == "sample-from-cluster-files":
        return _run_sample_from_cluster_files(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
