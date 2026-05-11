r"""Hidden module for submitting midsection round HPC jobs."""
import argparse
import math
import typing as tp
import os
from pathlib import Path
from datetime import datetime

from . import _config

__all__ = ["prepare_midsection_round_jobs"]


def _get_prev_round_buf_and_mol_idxs_files(
    path: Path,
    round_idx: int,
) -> list[tuple[Path, Path]]:
    """Get buffer and index files from previous round."""
    path = Path(path).resolve()
    buf_files = sorted(path.glob(f"round-{round_idx - 1}-bufs*.npy"))
    idx_files = sorted(path.glob(f"round-{round_idx - 1}-idxs*.pkl"))

    if len(buf_files) != len(idx_files):
        raise ValueError(
            f"Mismatch: found {len(buf_files)} bufs and {len(idx_files)} idxs for round {round_idx - 1}"
        )
    if len(buf_files) == 0:
        raise ValueError(f"No buffer/index files found for round {round_idx - 1} in {path}")

    return list(zip(buf_files, idx_files))


def _sort_batch(b: tp.Sequence[tuple[Path, Path]]) -> tuple[tuple[Path, Path], ...]:
    """Sort batch by uint bits (largest first) for better cluster quality."""
    return tuple(
        sorted(
            b,
            key=lambda b: int(b[0].name.split("uint")[-1].split(".")[0]) if "uint" in b[0].name else 0,
            reverse=True,
        )
    )


def _chunk_file_pairs_in_batches(
    file_pairs: tp.Sequence[tuple[Path, Path]],
    bin_size: int,
) -> list[tuple[str, tuple[tuple[Path, Path], ...]]]:
    """Chunk file pairs into batches."""
    from bblean.utils import batched

    z = len(str(math.ceil(len(file_pairs) / bin_size)))
    batches = [
        (str(i).zfill(z), _sort_batch(b))
        for i, b in enumerate(batched(file_pairs, bin_size))
    ]
    return batches


def _generate_midsection_round_script(
    batches: list[tuple[str, tuple[tuple[Path, Path], ...]]],
    output_dir: Path,
    round_idx: int,
    params: dict,
    slurm_params: dict,
    max_jobs_per_script: int = _config.MAX_JOBS_PER_SCRIPT,
) -> list[Path]:
    """Generate shell script(s) for midsection round job submission.

    If total batches > max_jobs_per_script, creates multiple scripts.
    Returns list of script paths.
    """
    output_dir = output_dir.resolve()

    # Split batches into chunks if needed
    script_paths = []
    num_scripts = (len(batches) + max_jobs_per_script - 1) // max_jobs_per_script

    for script_idx in range(num_scripts):
        start_batch = script_idx * max_jobs_per_script
        end_batch = min((script_idx + 1) * max_jobs_per_script, len(batches))
        batch_chunk = batches[start_batch:end_batch]

        if num_scripts > 1:
            script_path = output_dir / f"submit_midsection_round_{round_idx}_jobs_{script_idx + 1}.sh"
        else:
            script_path = output_dir / f"submit_midsection_round_{round_idx}_jobs.sh"

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n\n")

            for batch_label, batch_path_pairs in batch_chunk:
                job_name = f"midsection_{round_idx}_{batch_label}"
                out_log = output_dir / f"logs/midsection_{round_idx}_{batch_label}_%j.log"
                job_script = output_dir / f".job_midsection_{round_idx}_{batch_label}.sh"

                pairs_str = ",".join([f"{bp[0]}:{bp[1]}" for bp in batch_path_pairs])

                cmd = (
                    f"python -m iChem.bitbirch._hpc_midsection "
                    f"--output-dir {output_dir} "
                    f"--round-idx {round_idx} "
                    f"--batch-label {batch_label} "
                    f"--file-pairs {pairs_str} "
                    f"--threshold {params['threshold']} "
                    f"--branching-factor {params['branching_factor']} "
                    f"--merge-criterion {params['merge_criterion']} "
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
        script_paths.append(script_path)

    return script_paths


def prepare_midsection_round_jobs(
    output_dir: Path,
    round_idx: int,
    bin_size: int = _config.BIN_SIZE,
    threshold: float | None = None,
    branching_factor: int | None = None,
    merge_criterion: str = _config.MERGE_CRITERION,
    reclustering_iterations: int = _config.RECLUSTERING_ITERATIONS_MIDSECTION,
    reclustering_extra_threshold: float = _config.RECLUSTERING_EXTRA_THRESHOLD,
    # SLURM parameters
    slurm_mem: str = _config.SLURM_MEM_MIDSECTION,
    slurm_cpus: int = _config.SLURM_CPUS_MIDSECTION,
    slurm_time: str = _config.SLURM_TIME,
    slurm_partition: str = _config.SLURM_PARTITION,
    max_jobs_per_script: int = _config.MAX_JOBS_PER_SCRIPT,
    verbose: bool = False,
) -> list[Path]:
    r"""Prepare and generate midsection round job submission script.

    Parameters
    ----------
    output_dir : Path
        Output directory where previous round results are stored
    round_idx : int
        Current round index (will read from round_{round_idx-1} files)
    bin_size : int
        Number of buffer/index pairs per batch job
    threshold : float
        BitBirch threshold for this round
    branching_factor : int
        BitBirch branching factor for this round
    merge_criterion : str
        Merge criterion for BitBirch
    reclustering_iterations : int
        Number of reclustering iterations
    reclustering_extra_threshold : float
        Extra threshold for reclustering
    slurm_mem : str
        SLURM memory allocation
    slurm_cpus : int
        SLURM CPU count
    slurm_time : str
        SLURM time limit
    slurm_partition : str
        SLURM partition (optional)
    verbose : bool
        Print progress information

    Returns
    -------
    Path
        Path to generated submission script
    """
    if threshold is None:
        threshold = _config.THRESHOLD
    if branching_factor is None:
        branching_factor = _config.BRANCHING_FACTOR

    output_dir = Path(output_dir)
    (output_dir / "logs").mkdir(exist_ok=True)

    if verbose:
        print(f"[Midsection Round {round_idx} Setup]")
        print(f"  Output directory: {output_dir}")
        print(f"  Bin size: {bin_size}")
        print(f"  Reading from: round-{round_idx - 1}-* files")

    file_pairs = _get_prev_round_buf_and_mol_idxs_files(output_dir, round_idx)

    if verbose:
        print(f"  Found {len(file_pairs)} buffer/index file pairs")

    batches = _chunk_file_pairs_in_batches(file_pairs, bin_size)

    if verbose:
        print(f"  Created {len(batches)} batches")
        for label, batch_pairs in batches:
            print(f"    Batch {label}: {len(batch_pairs)} pairs")

    params = {
        "threshold": threshold,
        "branching_factor": branching_factor,
        "merge_criterion": merge_criterion,
        "reclustering_iterations": reclustering_iterations,
        "extra_threshold": reclustering_extra_threshold,
    }

    slurm_params = {
        "mem": slurm_mem,
        "cpus": slurm_cpus,
        "time": slurm_time,
        "partition": slurm_partition,
    }

    script_paths = _generate_midsection_round_script(
        batches,
        output_dir,
        round_idx,
        params,
        slurm_params,
        max_jobs_per_script,
    )

    if verbose:
        if len(script_paths) == 1:
            rel_path = os.path.relpath(script_paths[0], os.getcwd())
            print(f"\n✓ Generated submission script: {script_paths[0]}")
            print(f"  Run with: bash {rel_path}")
        else:
            print(f"\n✓ Generated {len(script_paths)} submission scripts (max {max_jobs_per_script} jobs per script):")
            for i, path in enumerate(script_paths, 1):
                rel_path = os.path.relpath(path, os.getcwd())
                print(f"  {i}. bash {rel_path}")

    return script_paths[0] if len(script_paths) == 1 else script_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare midsection round job submission script"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory containing previous round results"
    )
    parser.add_argument(
        "--round-idx",
        type=int,
        required=True,
        help="Current round index (will read round-{round_idx-1}-* files)"
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=_config.BIN_SIZE,
        help="Number of buffer/index pairs per batch job"
    )
    parser.add_argument("--threshold", type=float, default=None, help="BitBirch threshold")
    parser.add_argument("--branching-factor", type=int, default=None, help="BitBirch branching factor")
    parser.add_argument("--merge-criterion", type=str, default=_config.MERGE_CRITERION, help="Merge criterion")
    parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_MIDSECTION, help="Reclustering iterations")
    parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold")
    parser.add_argument("--slurm-mem", type=str, default=_config.SLURM_MEM_MIDSECTION, help="SLURM memory")
    parser.add_argument("--slurm-cpus", type=int, default=_config.SLURM_CPUS_MIDSECTION, help="SLURM CPUs")
    parser.add_argument("--slurm-time", type=str, default=_config.SLURM_TIME, help="SLURM time")
    parser.add_argument("--slurm-partition", type=str, default=_config.SLURM_PARTITION, help="SLURM partition")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    prepare_midsection_round_jobs(
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
        verbose=args.verbose,
    )
