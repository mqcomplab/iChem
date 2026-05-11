r"""Hidden module for submitting initial round HPC jobs."""
import argparse
import math
import typing as tp
import os
from pathlib import Path
from datetime import datetime
import gzip as gz

from . import _config

__all__ = ["prepare_initial_round_jobs"]


def _get_file_length(file_path: Path) -> int:
    """Get line count from .smi or .smi.gz file efficiently."""
    file_path = Path(file_path)
    name = file_path.name

    if name.endswith('.smi.gz'):
        with gz.open(file_path, 'rt') as f:
            return sum(1 for _ in f)
    elif name.endswith('.smi'):
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    else:
        raise ValueError('Only .smi and .smi.gz files are supported')


def _chunk_smi_files(
    smi_files: tp.Sequence[Path],
    files_per_job: int,
) -> list[tuple[str, list[Path], int, int]]:
    """Group SMI files into batches with global molecule indices.

    Returns list of (label, file_list, start_idx, end_idx) tuples
    """
    batches = []
    z = len(str(math.ceil(len(smi_files) / files_per_job)))
    global_idx = 0

    for batch_idx, batch_start in enumerate(range(0, len(smi_files), files_per_job)):
        batch_end = min(batch_start + files_per_job, len(smi_files))
        file_batch = smi_files[batch_start:batch_end]

        batch_mol_count = sum(_get_file_length(f) for f in file_batch)
        start_idx = global_idx
        end_idx = global_idx + batch_mol_count

        label = str(batch_idx).zfill(z)
        batches.append((label, file_batch, start_idx, end_idx))

        global_idx = end_idx

    return batches


def _generate_initial_round_script(
    batches: list[tuple[str, list[Path], int, int]],
    output_dir: Path,
    params: dict,
    slurm_params: dict,
    result_base_dir: Path | None = None,
) -> Path:
    """Generate shell script for initial round job submission."""
    output_dir = output_dir.resolve()
    script_path = output_dir / "submit_initial_jobs.sh"

    result_dir_arg = f" --output-dir {result_base_dir.resolve()}" if result_base_dir else ""

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")

        for label, file_batch, start_idx, end_idx in batches:
            job_name = f"initial_{label}"
            out_log = output_dir / f"logs/initial_{label}_%j.log"
            job_script = output_dir / f".job_initial_{label}.sh"

            files_str = " ".join(str(f) for f in file_batch)

            cmd = (
                f"python -m iChem.bitbirch._hpc_initial "
                f"--smi-files {files_str} "
                f"--start-idx {start_idx} --end-idx {end_idx} "
                f"--label {label} "
                f"--output-dir {output_dir} "
                f"--threshold {params['threshold']} "
                f"--branching-factor {params['branching_factor']} "
                f"--merge-criterion {params['merge_criterion']} "
                f"--fp-type {params['fp_type']} "
                f"--n-bits {params['n_bits']} "
                f"--reclustering-iterations {params['reclustering_iterations']} "
                f"--extra-threshold {params['extra_threshold']}"
            )

            f.write(f"cat > {job_script} <<'JOBEOF'\n")
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write(f"#SBATCH --output={out_log}\n")
            f.write(f"#SBATCH --mem={slurm_params['mem']}\n")
            f.write(f"#SBATCH --cpus-per-task={slurm_params['cpus']}\n")
            f.write(f"#SBATCH --time={slurm_params['time']}\n")
            if slurm_params.get('partition'):
                f.write(f"#SBATCH --partition={slurm_params['partition']}\n")
            f.write("\n")
            f.write("# Load conda module\n")
            f.write("module load conda\n\n")
            f.write("# Activate iChem environment\n")
            f.write("conda activate iChem\n\n")
            f.write(f"{cmd}\n")
            f.write("JOBEOF\n")
            f.write(f"sbatch {job_script}\n")
            f.write(f"rm {job_script}\n\n")

    script_path.chmod(0o755)
    return script_path


def prepare_initial_round_jobs(
    input_files: tp.Sequence[Path],
    output_dir: Path,
    files_per_job: int = _config.FILES_PER_JOB,
    threshold: float = _config.THRESHOLD,
    branching_factor: int = _config.BRANCHING_FACTOR,
    merge_criterion: str = _config.MERGE_CRITERION,
    fp_type: str = _config.FINGERPRINT_TYPE,
    n_bits: int = _config.N_BITS,
    reclustering_iterations: int = _config.RECLUSTERING_ITERATIONS_INITIAL,
    extra_threshold: float = _config.RECLUSTERING_EXTRA_THRESHOLD,
    # SLURM parameters
    slurm_mem: str = _config.SLURM_MEM_INITIAL,
    slurm_cpus: int = _config.SLURM_CPUS_INITIAL,
    slurm_time: str = _config.SLURM_TIME,
    slurm_partition: str = _config.SLURM_PARTITION,
    result_base_dir: Path | None = None,
    verbose: bool = False,
) -> Path:
    r"""Prepare and generate initial round job submission script.

    Parameters
    ----------
    input_files : Sequence[Path]
        SMILES files (.smi or .smi.gz) to cluster
    output_dir : Path
        Output directory for submission script and logs
    files_per_job : int
        Number of SMILES files to process per initial job
    threshold : float
        BitBirch threshold
    branching_factor : int
        BitBirch branching factor
    merge_criterion : str
        Merge criterion for BitBirch
    fp_type : str
        Fingerprint type
    n_bits : int
        Number of fingerprint bits
    reclustering_iterations : int
        Number of reclustering iterations
    extra_threshold : float
        Extra threshold for reclustering
    slurm_mem : str
        SLURM memory allocation
    slurm_cpus : int
        SLURM CPU count
    slurm_time : str
        SLURM time limit
    slurm_partition : str
        SLURM partition (optional)
    result_base_dir : Path, optional
        Base directory for results. If None, each job creates its own bb_multiround_results/{id}
    verbose : bool
        Print progress information

    Returns
    -------
    Path
        Path to generated submission script
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    input_files = sorted([Path(f) for f in input_files])

    if verbose:
        print(f"[Initial Round Setup]")
        print(f"  Input files: {len(input_files)}")
        print(f"  Files per job: {files_per_job}")
        print(f"  Output directory: {output_dir}")

    batches = _chunk_smi_files(input_files, files_per_job)

    if verbose:
        print(f"  Created {len(batches)} jobs")
        for label, file_batch, start_idx, end_idx in batches:
            print(f"    Job {label}: {len(file_batch)} files, molecules {start_idx}-{end_idx}")

    params = {
        "threshold": threshold,
        "branching_factor": branching_factor,
        "merge_criterion": merge_criterion,
        "fp_type": fp_type,
        "n_bits": n_bits,
        "reclustering_iterations": reclustering_iterations,
        "extra_threshold": extra_threshold,
    }

    slurm_params = {
        "mem": slurm_mem,
        "cpus": slurm_cpus,
        "time": slurm_time,
        "partition": slurm_partition,
    }

    script_path = _generate_initial_round_script(
        batches,
        output_dir,
        params,
        slurm_params,
        result_base_dir,
    )

    if verbose:
        rel_path = os.path.relpath(script_path, os.getcwd())
        print(f"\n✓ Generated submission script: {script_path}")
        print(f"  Run with: bash {rel_path}")

    return script_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare initial round job submission script"
    )
    parser.add_argument("--input-files", nargs="+", required=True, help="Input SMILES files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for submission script and logs"
    )
    parser.add_argument(
        "--files-per-job",
        type=int,
        default=_config.FILES_PER_JOB,
        help="Number of SMILES files per initial job"
    )
    parser.add_argument("--threshold", type=float, default=_config.THRESHOLD, help="BitBirch threshold")
    parser.add_argument("--branching-factor", type=int, default=_config.BRANCHING_FACTOR, help="BitBirch branching factor")
    parser.add_argument("--merge-criterion", type=str, default=_config.MERGE_CRITERION, help="Merge criterion")
    parser.add_argument("--fp-type", type=str, default=_config.FINGERPRINT_TYPE, help="Fingerprint type")
    parser.add_argument("--n-bits", type=int, default=_config.N_BITS, help="Number of bits")
    parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_INITIAL, help="Reclustering iterations")
    parser.add_argument("--extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold")
    parser.add_argument("--slurm-mem", type=str, default=_config.SLURM_MEM_INITIAL, help="SLURM memory")
    parser.add_argument("--slurm-cpus", type=int, default=_config.SLURM_CPUS_INITIAL, help="SLURM CPUs")
    parser.add_argument("--slurm-time", type=str, default=_config.SLURM_TIME, help="SLURM time")
    parser.add_argument("--slurm-partition", type=str, default=_config.SLURM_PARTITION, help="SLURM partition")
    parser.add_argument(
        "--result-base-dir",
        type=Path,
        default=None,
        help="Base directory for results (optional)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    prepare_initial_round_jobs(
        input_files=args.input_files,
        output_dir=args.output_dir,
        files_per_job=args.files_per_job,
        threshold=args.threshold,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        fp_type=args.fp_type,
        n_bits=args.n_bits,
        reclustering_iterations=args.reclustering_iterations,
        extra_threshold=args.extra_threshold,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        result_base_dir=args.result_base_dir,
        verbose=args.verbose,
    )
