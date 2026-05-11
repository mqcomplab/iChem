r"""Hidden module for submitting final round HPC job."""
import argparse
from pathlib import Path
from datetime import datetime

from . import _config

__all__ = ["prepare_final_round_job"]


def _get_prev_round_buf_and_mol_idxs_files(
    path: Path,
    round_idx: int,
) -> list[tuple[Path, Path]]:
    """Get buffer and index files from previous round."""
    path = Path(path).resolve()
    buf_files = sorted(path.glob(f"round-{round_idx}-bufs*.npy"))
    idx_files = sorted(path.glob(f"round-{round_idx}-idxs*.pkl"))

    if len(buf_files) != len(idx_files):
        raise ValueError(
            f"Mismatch: found {len(buf_files)} bufs and {len(idx_files)} idxs for round {round_idx}"
        )
    if len(buf_files) == 0:
        raise ValueError(f"No buffer/index files found for round {round_idx} in {path}")

    return list(zip(buf_files, idx_files))


def _generate_final_round_script(
    output_dir: Path,
    prev_round_idx: int,
    file_pairs: list[tuple[Path, Path]],
    params: dict,
    slurm_params: dict,
) -> Path:
    """Generate shell script for final round job submission."""
    output_dir = output_dir.resolve()
    script_path = output_dir / "submit_final_round_job.sh"

    job_name = "final_round"
    out_log = output_dir / "logs/final_round_%j.log"
    job_script = output_dir / ".job_final_round.sh"

    pairs_str = ",".join([f"{fp[0]}:{fp[1]}" for fp in file_pairs])

    cmd = (
        f"python -m iChem.bitbirch._hpc_final "
        f"--output-dir {output_dir} "
        f"--round-idx {prev_round_idx} "
        f"--file-pairs {pairs_str} "
        f"--threshold {params['threshold']} "
        f"--branching-factor {params['branching_factor']} "
        f"--merge-criterion {params['merge_criterion']} "
        f"--reclustering-iterations {params['reclustering_iterations']} "
        f"--extra-threshold {params['extra_threshold']} "
    )

    if params['save_tree']:
        cmd += "--save-tree "
    if params['save_centroids']:
        cmd += "--save-centroids "

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
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
        f.write(f"rm {job_script}\n")

    script_path.chmod(0o755)
    return script_path


def prepare_final_round_job(
    output_dir: Path,
    prev_round_idx: int,
    threshold: float | None = None,
    branching_factor: int | None = None,
    merge_criterion: str = _config.MERGE_CRITERION,
    reclustering_iterations: int = _config.RECLUSTERING_ITERATIONS_FINAL,
    reclustering_extra_threshold: float = _config.RECLUSTERING_EXTRA_THRESHOLD,
    save_tree: bool = False,
    save_centroids: bool = True,
    # SLURM parameters
    slurm_mem: str = _config.SLURM_MEM_FINAL,
    slurm_cpus: int = _config.SLURM_CPUS_FINAL,
    slurm_time: str = _config.SLURM_TIME,
    slurm_partition: str = _config.SLURM_PARTITION,
    verbose: bool = False,
) -> Path:
    r"""Prepare and generate final round job submission script.

    Parameters
    ----------
    output_dir : Path
        Output directory where previous round results are stored
    prev_round_idx : int
        Previous round index (will read from round_{prev_round_idx} files)
    threshold : float
        BitBirch threshold for final round
    branching_factor : int
        BitBirch branching factor for final round
    merge_criterion : str
        Merge criterion for BitBirch
    reclustering_iterations : int
        Number of reclustering iterations
    reclustering_extra_threshold : float
        Extra threshold for reclustering
    save_tree : bool
        Whether to save the final BitBirch tree
    save_centroids : bool
        Whether to save centroids and cluster assignments
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
        print(f"[Final Round Setup]")
        print(f"  Output directory: {output_dir}")
        print(f"  Reading from: round-{prev_round_idx}-* files")

    file_pairs = _get_prev_round_buf_and_mol_idxs_files(output_dir, prev_round_idx)

    if verbose:
        print(f"  Found {len(file_pairs)} buffer/index file pairs to merge")

    params = {
        "threshold": threshold,
        "branching_factor": branching_factor,
        "merge_criterion": merge_criterion,
        "reclustering_iterations": reclustering_iterations,
        "extra_threshold": reclustering_extra_threshold,
        "save_tree": save_tree,
        "save_centroids": save_centroids,
    }

    slurm_params = {
        "mem": slurm_mem,
        "cpus": slurm_cpus,
        "time": slurm_time,
        "partition": slurm_partition,
    }

    script_path = _generate_final_round_script(
        output_dir,
        prev_round_idx,
        file_pairs,
        params,
        slurm_params,
    )

    if verbose:
        print(f"\n✓ Generated submission script: {script_path}")
        print(f"  Run with: ./{script_path.name}")

    return script_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare final round job submission script"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory containing previous round results"
    )
    parser.add_argument(
        "--prev-round-idx",
        type=int,
        required=True,
        help="Previous round index (will read round-{prev_round_idx}-* files)"
    )
    parser.add_argument("--threshold", type=float, default=None, help="BitBirch threshold")
    parser.add_argument("--branching-factor", type=int, default=None, help="BitBirch branching factor")
    parser.add_argument("--merge-criterion", type=str, default=_config.MERGE_CRITERION, help="Merge criterion")
    parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_FINAL, help="Reclustering iterations")
    parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold")
    parser.add_argument("--save-tree", action="store_true", default=False, help="Save the BitBirch tree")
    parser.add_argument("--save-centroids", action="store_true", default=True, help="Save centroids and cluster assignments")
    parser.add_argument("--slurm-mem", type=str, default=_config.SLURM_MEM_FINAL, help="SLURM memory")
    parser.add_argument("--slurm-cpus", type=int, default=_config.SLURM_CPUS_FINAL, help="SLURM CPUs")
    parser.add_argument("--slurm-time", type=str, default=_config.SLURM_TIME, help="SLURM time")
    parser.add_argument("--slurm-partition", type=str, default=_config.SLURM_PARTITION, help="SLURM partition")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    prepare_final_round_job(
        output_dir=args.output_dir,
        prev_round_idx=args.prev_round_idx,
        threshold=args.threshold,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        reclustering_iterations=args.reclustering_iterations,
        reclustering_extra_threshold=args.reclustering_extra_threshold,
        save_tree=args.save_tree,
        save_centroids=args.save_centroids,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        verbose=args.verbose,
    )
