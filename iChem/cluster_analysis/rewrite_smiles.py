from pathlib import Path
import pickle as pkl
import argparse
import gzip
import re
import sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np

_clusters_shared = None
_smi_files_shared = None
_input_smiles_dir_shared = None
_output_dir_shared = None
_smiles_per_file_shared = None
_compressed_shared = None
_write_database_ids_shared = None
_cluster_files_shared = None

_CLUSTER_FILE_RE = re.compile(r"cluster_(\d+)\.npy$")


def _cluster_output_path(output_dir: str, cluster_id: int, compressed: bool) -> Path:
    output_path = Path(output_dir)
    ext = '.smi.gz' if compressed else '.smi'
    return output_path / f"cluster_{cluster_id}{ext}"


def _cluster_database_ids_output_path(output_dir: str, cluster_id: int, compressed: bool) -> Path:
    output_path = Path(output_dir)
    ext = '.txt.gz' if compressed else '.txt'
    return output_path / f"cluster_{cluster_id}_database_ids{ext}"


def _cluster_id_from_path(cluster_file: Path) -> int:
    match = _CLUSTER_FILE_RE.fullmatch(cluster_file.name)
    if match is None:
        raise ValueError(f"Invalid cluster file name: {cluster_file.name}. Expected cluster_<id>.npy")
    return int(match.group(1))


def _parse_smiles_and_database_id(line: str) -> tuple[str, str | None]:
    token = line.strip()
    if not token:
        return "", None

    # Common formats: "SMILES,ID", "SMILES<TAB>ID", "SMILES ID"
    if "\t" in token:
        smiles_field, id_field = token.split("\t", 1)
        smiles = smiles_field.strip().split(" ", 1)[0].strip()
        database_id = id_field.strip().split(" ", 1)[0].strip() if id_field.strip() else None
        return smiles, database_id

    if "," in token:
        smiles, database_id = token.rsplit(",", 1)
        smiles = smiles.strip()
        database_id = database_id.strip()
        return smiles, database_id if database_id else None

    parts = token.split()
    smiles = parts[0].strip()
    database_id = parts[1].strip() if len(parts) > 1 else None
    return smiles, database_id


def _load_smiles_and_database_ids(file_path: Path, include_database_ids: bool) -> tuple[list[str], list[str] | None]:
    smiles_list: list[str] = []
    database_ids: list[str] | None = [] if include_database_ids else None

    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            smiles, database_id = _parse_smiles_and_database_id(line)
            smiles_list.append(smiles)
            if include_database_ids:
                database_ids.append(database_id or "")

    return smiles_list, database_ids


def _load_cluster_entries(sorted_indices, smi_files, smiles_per_file, write_database_ids):
    cluster_smiles = []
    cluster_database_ids = [] if write_database_ids else None

    current_file_id = -1
    current_file_smiles = None
    file_start = 0

    for idx in sorted_indices:
        file_id = int(idx) // smiles_per_file

        if file_id != current_file_id:
            current_file_smiles = None

            if file_id >= len(smi_files):
                raise IndexError(f"File ID {file_id} out of range (only {len(smi_files)} files)")

            smi_file = smi_files[file_id]
            current_file_smiles, current_file_database_ids = _load_smiles_and_database_ids(
                smi_file,
                include_database_ids=write_database_ids,
            )

            current_file_id = file_id
            file_start = file_id * smiles_per_file

        pos_in_file = int(idx) - file_start
        cluster_smiles.append(current_file_smiles[pos_in_file])
        if write_database_ids and cluster_database_ids is not None:
            if current_file_database_ids is None:
                cluster_database_ids.append("")
            else:
                cluster_database_ids.append(current_file_database_ids[pos_in_file])

    return cluster_smiles, cluster_database_ids


def _init_worker(clusters, smi_files, input_smiles_dir, output_dir, smiles_per_file, compressed, write_database_ids):
    global _clusters_shared, _smi_files_shared, _input_smiles_dir_shared, _output_dir_shared, _smiles_per_file_shared, _compressed_shared, _write_database_ids_shared
    _clusters_shared = clusters
    _smi_files_shared = smi_files
    _input_smiles_dir_shared = input_smiles_dir
    _output_dir_shared = output_dir
    _smiles_per_file_shared = smiles_per_file
    _compressed_shared = compressed
    _write_database_ids_shared = write_database_ids


def _process_cluster_worker(cluster_id):
    """Worker function to process a single cluster."""
    output_path = Path(_output_dir_shared)
    output_file = _cluster_output_path(str(output_path), cluster_id, _compressed_shared)
    database_ids_output_file = _cluster_database_ids_output_path(str(output_path), cluster_id, _compressed_shared)

    # Skip if already processed
    if output_file.exists() and (not _write_database_ids_shared or database_ids_output_file.exists()):
        print(f"Cluster {cluster_id}: already exists, skipping", flush=True)
        sys.stdout.flush()
        return cluster_id

    cluster_indices = _clusters_shared[cluster_id]
    if not cluster_indices:
        return cluster_id

    sorted_indices = sorted(cluster_indices)
    cluster_smiles, cluster_database_ids = _load_cluster_entries(
        sorted_indices,
        _smi_files_shared,
        _smiles_per_file_shared,
        _write_database_ids_shared,
    )

    if _compressed_shared:
        with gzip.open(output_file, 'wt') as f:
            for smi in cluster_smiles:
                f.write(smi + '\n')
    else:
        with open(output_file, 'w') as f:
            for smi in cluster_smiles:
                f.write(smi + '\n')

    if _write_database_ids_shared and cluster_database_ids is not None:
        if _compressed_shared:
            with gzip.open(database_ids_output_file, 'wt') as f:
                for database_id in cluster_database_ids:
                    f.write(database_id + '\n')
        else:
            with open(database_ids_output_file, 'w') as f:
                for database_id in cluster_database_ids:
                    f.write(database_id + '\n')

    print(f"Cluster {cluster_id}: wrote {len(cluster_smiles)} molecules to {output_file}", flush=True)
    sys.stdout.flush()

    return cluster_id


def _init_npy_worker(smi_files, input_smiles_dir, output_dir, smiles_per_file, compressed, write_database_ids):
    global _clusters_shared, _cluster_files_shared, _smi_files_shared, _input_smiles_dir_shared, _output_dir_shared, _smiles_per_file_shared, _compressed_shared, _write_database_ids_shared
    _clusters_shared = None
    _cluster_files_shared = None
    _smi_files_shared = smi_files
    _input_smiles_dir_shared = input_smiles_dir
    _output_dir_shared = output_dir
    _smiles_per_file_shared = smiles_per_file
    _compressed_shared = compressed
    _write_database_ids_shared = write_database_ids


def _process_cluster_npy_worker(cluster_file):
    """Worker function to process a single cluster .npy file."""
    cluster_file = Path(cluster_file)
    cluster_id = _cluster_id_from_path(cluster_file)

    output_path = Path(_output_dir_shared)
    output_file = _cluster_output_path(str(output_path), cluster_id, _compressed_shared)
    database_ids_output_file = _cluster_database_ids_output_path(str(output_path), cluster_id, _compressed_shared)

    if output_file.exists() and (not _write_database_ids_shared or database_ids_output_file.exists()):
        print(f"Cluster {cluster_id}: already exists, skipping", flush=True)
        sys.stdout.flush()
        return cluster_id

    cluster_indices = np.load(cluster_file, mmap_mode="r")
    if len(cluster_indices) == 0:
        return cluster_id

    sorted_indices = np.sort(cluster_indices)
    cluster_smiles, cluster_database_ids = _load_cluster_entries(
        sorted_indices,
        _smi_files_shared,
        _smiles_per_file_shared,
        _write_database_ids_shared,
    )

    if _compressed_shared:
        with gzip.open(output_file, 'wt') as f:
            for smi in cluster_smiles:
                f.write(smi + '\n')
    else:
        with open(output_file, 'w') as f:
            for smi in cluster_smiles:
                f.write(smi + '\n')

    if _write_database_ids_shared and cluster_database_ids is not None:
        if _compressed_shared:
            with gzip.open(database_ids_output_file, 'wt') as f:
                for database_id in cluster_database_ids:
                    f.write(database_id + '\n')
        else:
            with open(database_ids_output_file, 'w') as f:
                for database_id in cluster_database_ids:
                    f.write(database_id + '\n')

    print(f"Cluster {cluster_id}: wrote {len(cluster_smiles)} molecules to {output_file}", flush=True)
    sys.stdout.flush()

    return cluster_id


def find_first_missing_cluster(output_dir: str,
                               num_clusters: int,
                               compressed: bool = False,
                               write_database_ids: bool = False) -> int:
    """Find the first cluster that hasn't been written yet.

    Parameters
    ----------
    output_dir : str
        Directory containing output SMILES files.
    num_clusters : int
        Total number of clusters.
    compressed : bool
        Whether to look for .smi.gz or .smi files.

    Returns
    -------
    int
        Index of first missing cluster, or num_clusters if all exist.
    """
    output_path = Path(output_dir)

    for cluster_id in range(num_clusters):
        output_file = _cluster_output_path(str(output_path), cluster_id, compressed)
        database_ids_output_file = _cluster_database_ids_output_path(str(output_path), cluster_id, compressed)
        smiles_missing = not output_file.exists()
        database_ids_missing = write_database_ids and not database_ids_output_file.exists()
        if smiles_missing or database_ids_missing:
            return cluster_id

    return num_clusters


def find_first_missing_cluster_from_files(output_dir: str,
                                          cluster_files: list[Path],
                                          compressed: bool = False,
                                          write_database_ids: bool = False) -> int:
    output_path = Path(output_dir)

    for cluster_file in cluster_files:
        cluster_id = _cluster_id_from_path(Path(cluster_file))
        output_file = _cluster_output_path(str(output_path), cluster_id, compressed)
        database_ids_output_file = _cluster_database_ids_output_path(str(output_path), cluster_id, compressed)
        smiles_missing = not output_file.exists()
        database_ids_missing = write_database_ids and not database_ids_output_file.exists()
        if smiles_missing or database_ids_missing:
            return cluster_id

    if not cluster_files:
        return 0

    return _cluster_id_from_path(Path(cluster_files[-1])) + 1


def rewrite_smiles_by_cluster(clusters: list[list[int]],
                               input_smiles_dir: str,
                               output_dir: str,
                               smiles_per_file: int = 1_000_000,
                               compressed: bool = False,
                               num_workers: int = 8,
                               start_at: int = 0,
                               write_database_ids: bool = False):
    """Reorganize SMILES files by cluster, loading only necessary molecules.

    Parameters
    ----------
    clusters : list[list[int]]
        List of clusters where each cluster is a list of SMILES indices.
    input_smiles_dir : str
        Directory containing input SMILES files (*.smi or *.smi.gz).
    output_dir : str
        Directory to write cluster-organized SMILES files.
    smiles_per_file : int
        Number of SMILES per input file (default 1M).
    compressed : bool
        If True, write gzipped files (.smi.gz), else plain text (.smi).
    num_workers : int
        Number of parallel processes (default 8).
    start_at : int
        Cluster index to start processing from (default 0). Clusters before this index are freed from memory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_smiles_dir)
    smi_files = sorted(input_dir.glob("*.smi")) + sorted(input_dir.glob("*.smi.gz"))

    if not smi_files:
        raise FileNotFoundError(f"No SMILES files found in {input_smiles_dir}")

    # Free memory for already-processed clusters
    for i in range(start_at):
        clusters[i] = None

    print(f"Starting processing from cluster {start_at}", flush=True)

    # Process clusters in parallel
    with Pool(processes=num_workers,
              initializer=_init_worker,
              initargs=(clusters, smi_files, input_smiles_dir, output_dir, smiles_per_file, compressed, write_database_ids)) as pool:
        for cluster_id in pool.imap_unordered(_process_cluster_worker, range(start_at, len(clusters))):
            # Memory released after each cluster is processed
            clusters[cluster_id] = None


def rewrite_smiles_by_cluster_from_npy_dir(clusters_dir: str,
                                           input_smiles_dir: str,
                                           output_dir: str,
                                           smiles_per_file: int = 1_000_000,
                                           compressed: bool = False,
                                           num_workers: int = 8,
                                           start_at: int = 0,
                                           write_database_ids: bool = False):
    """Reorganize SMILES files by cluster using one .npy file per cluster."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_smiles_dir)
    smi_files = sorted(input_dir.glob("*.smi")) + sorted(input_dir.glob("*.smi.gz"))

    if not smi_files:
        raise FileNotFoundError(f"No SMILES files found in {input_smiles_dir}")

    cluster_path = Path(clusters_dir)
    cluster_files = sorted(cluster_path.glob("cluster_*.npy"), key=_cluster_id_from_path)

    if not cluster_files:
        raise FileNotFoundError(f"No cluster_*.npy files found in {clusters_dir}")

    if output_path.exists() and start_at == 0:
        first_missing = find_first_missing_cluster_from_files(
            str(output_dir),
            cluster_files,
            compressed,
            write_database_ids,
        )
        last_cluster_id = _cluster_id_from_path(cluster_files[-1])
        if first_missing <= last_cluster_id:
            print(f"Output directory exists. Found clusters up to {first_missing - 1} completed.", flush=True)
            print(f"Resuming from cluster {first_missing}", flush=True)
            start_at = first_missing

    cluster_files = [cluster_file for cluster_file in cluster_files if _cluster_id_from_path(cluster_file) >= start_at]

    if not cluster_files:
        return

    print(f"Starting processing from cluster {start_at}", flush=True)

    with Pool(processes=num_workers,
              initializer=_init_npy_worker,
              initargs=(smi_files, input_smiles_dir, output_dir, smiles_per_file, compressed, write_database_ids)) as pool:
        for _ in pool.imap_unordered(_process_cluster_npy_worker, cluster_files):
            pass


def rewrite_single_cluster_from_npy(cluster_file: str,
                                    input_smiles_dir: str,
                                    output_dir: str,
                                    smiles_per_file: int = 1_000_000,
                                    compressed: bool = False,
                                    write_database_ids: bool = False,
                                    overwrite: bool = False) -> int:
    """Reorganize one cluster_<id>.npy file into a cluster SMILES file."""
    cluster_path = Path(cluster_file)
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    cluster_id = _cluster_id_from_path(cluster_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = _cluster_output_path(str(output_path), cluster_id, compressed)
    database_ids_output_file = _cluster_database_ids_output_path(str(output_path), cluster_id, compressed)

    if not overwrite and output_file.exists() and (not write_database_ids or database_ids_output_file.exists()):
        print(f"Cluster {cluster_id}: already exists, skipping", flush=True)
        return cluster_id

    input_dir = Path(input_smiles_dir)
    smi_files = sorted(input_dir.glob("*.smi")) + sorted(input_dir.glob("*.smi.gz"))
    if not smi_files:
        raise FileNotFoundError(f"No SMILES files found in {input_smiles_dir}")

    cluster_indices = np.load(cluster_path, mmap_mode="r")
    if len(cluster_indices) == 0:
        if compressed:
            with gzip.open(output_file, "wt"):
                pass
            if write_database_ids:
                with gzip.open(database_ids_output_file, "wt"):
                    pass
        else:
            output_file.touch()
            if write_database_ids:
                database_ids_output_file.touch()
        return cluster_id

    sorted_indices = np.sort(cluster_indices)
    cluster_smiles, cluster_database_ids = _load_cluster_entries(
        sorted_indices,
        smi_files,
        smiles_per_file,
        write_database_ids,
    )

    if compressed:
        with gzip.open(output_file, "wt") as f:
            for smi in cluster_smiles:
                f.write(smi + "\n")
    else:
        with open(output_file, "w") as f:
            for smi in cluster_smiles:
                f.write(smi + "\n")

    if write_database_ids and cluster_database_ids is not None:
        if compressed:
            with gzip.open(database_ids_output_file, "wt") as f:
                for database_id in cluster_database_ids:
                    f.write(database_id + "\n")
        else:
            with open(database_ids_output_file, "w") as f:
                for database_id in cluster_database_ids:
                    f.write(database_id + "\n")

    print(f"Cluster {cluster_id}: wrote {len(cluster_smiles)} molecules to {output_file}", flush=True)
    if write_database_ids:
        print(f"Cluster {cluster_id}: wrote database IDs to {database_ids_output_file}", flush=True)
    return cluster_id


def prepare_rewrite_smiles_npy_jobs(clusters_dir: str,
                                    input_smiles_dir: str,
                                    output_dir: str,
                                    submit_dir: str | None = None,
                                    smiles_per_file: int = 1_000_000,
                                    compressed: bool = True,
                                    write_database_ids: bool = False,
                                    overwrite: bool = False,
                                    max_jobs_per_script: int = 250,
                                    slurm_mem: str = "32G",
                                    slurm_cpus: int = 2,
                                    slurm_time: str = "24:00:00",
                                    slurm_partition: str | None = None,
                                    conda_env: str = "iChem") -> Path | list[Path]:
    """Generate shell script(s) that submit one SLURM job per cluster_<id>.npy file."""
    clusters_path = Path(clusters_dir)
    if not clusters_path.exists():
        raise FileNotFoundError(f"Cluster directory not found: {clusters_dir}")

    cluster_files = sorted(clusters_path.glob("cluster_*.npy"), key=_cluster_id_from_path)
    if not cluster_files:
        raise FileNotFoundError(f"No cluster_*.npy files found in {clusters_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    submit_path = Path(submit_dir) if submit_dir else output_path
    submit_path.mkdir(parents=True, exist_ok=True)
    logs_path = submit_path / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)

    if max_jobs_per_script < 1:
        raise ValueError("max_jobs_per_script must be >= 1")

    jobs = []
    for cluster_file in cluster_files:
        cluster_id = _cluster_id_from_path(cluster_file)
        output_file = _cluster_output_path(str(output_path), cluster_id, compressed)
        database_ids_output_file = _cluster_database_ids_output_path(str(output_path), cluster_id, compressed)

        if (not overwrite and output_file.exists()
                and (not write_database_ids or database_ids_output_file.exists())):
            continue

        jobs.append((cluster_id, cluster_file))

    if not jobs:
        script_path = submit_path / "submit_rewrite_smiles_jobs.sh"
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Total jobs: 0\n")
            f.write("echo 'No rewrite jobs to submit.'\n")
        script_path.chmod(0o755)
        return script_path

    script_paths: list[Path] = []
    num_scripts = (len(jobs) + max_jobs_per_script - 1) // max_jobs_per_script

    for script_idx in range(num_scripts):
        start_job = script_idx * max_jobs_per_script
        end_job = min((script_idx + 1) * max_jobs_per_script, len(jobs))
        job_chunk = jobs[start_job:end_job]

        if num_scripts == 1:
            script_path = submit_path / "submit_rewrite_smiles_jobs.sh"
        else:
            script_path = submit_path / f"submit_rewrite_smiles_jobs_{script_idx + 1}.sh"

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total jobs: {len(jobs)}\n")
            f.write(f"# Jobs in this script: {len(job_chunk)}\n")
            f.write(f"# Script {script_idx + 1} of {num_scripts}\n\n")

            for cluster_id, cluster_file in job_chunk:
                job_name = f"rewrite_cluster_{cluster_id}"
                out_log = logs_path / f"rewrite_cluster_{cluster_id}_%j.log"
                job_script = submit_path / f".job_rewrite_cluster_{cluster_id}.sh"

                cmd = (
                    "python -m iChem.cli rewrite-smiles-single-cluster-npy "
                    f"--cluster-file {cluster_file.resolve()} "
                    f"--smiles-dir {Path(input_smiles_dir).resolve()} "
                    f"--output-dir {output_path.resolve()} "
                    f"--smiles-per-file {smiles_per_file}"
                )
                if compressed:
                    cmd += " --compressed"
                if write_database_ids:
                    cmd += " --write-database-ids"
                if overwrite:
                    cmd += " --overwrite"

                f.write(f"cat > {job_script} <<'JOBEOF'\n")
                f.write("#!/bin/bash\n")
                f.write(f"#SBATCH --job-name={job_name}\n")
                f.write(f"#SBATCH --output={out_log}\n")
                f.write(f"#SBATCH --mem={slurm_mem}\n")
                f.write(f"#SBATCH --cpus-per-task={slurm_cpus}\n")
                f.write(f"#SBATCH --time={slurm_time}\n")
                if slurm_partition:
                    f.write(f"#SBATCH --partition={slurm_partition}\n")
                f.write("\n")
                f.write("module load conda\n")
                f.write(f"conda activate {conda_env}\n")
                f.write(f"{cmd}\n")
                f.write("JOBEOF\n")
                f.write(f"sbatch {job_script}\n")
                f.write(f"rm {job_script}\n\n")

        script_path.chmod(0o755)
        script_paths.append(script_path)

    return script_paths[0] if len(script_paths) == 1 else script_paths


def main():
    parser = argparse.ArgumentParser(description='Rewrite SMILES by cluster.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--clusters-file', type=str, help='Path to pickled clusters file.')
    group.add_argument('--clusters-dir', type=str, help='Directory containing cluster_<id>.npy files.')
    group.add_argument('--cluster-file', type=str, help='Path to one cluster_<id>.npy file.')
    parser.add_argument('--smiles-dir', type=str, required=True, help='Directory containing original SMILES files.')
    parser.add_argument('--smiles-per-file', type=int, default=1_000_000, help='Number of SMILES per input file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save cluster SMILES files.')
    parser.add_argument('--compressed', action='store_true', help='Write gzipped files.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel processes (default 8).')
    parser.add_argument('--start-at', type=int, default=0, help='Cluster index to start processing from (default 0).')
    parser.add_argument('--write-database-ids', action='store_true', help='Write ordered database IDs per cluster.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files for a cluster.')

    args = parser.parse_args()

    if args.clusters_file:
        with open(args.clusters_file, 'rb') as f:
            clusters = pkl.load(f)

        rewrite_smiles_by_cluster(clusters, args.smiles_dir, args.output_dir,
                                  args.smiles_per_file, args.compressed, args.num_workers, args.start_at,
                                  args.write_database_ids)
    elif args.clusters_dir:
        rewrite_smiles_by_cluster_from_npy_dir(args.clusters_dir, args.smiles_dir, args.output_dir,
                                               args.smiles_per_file, args.compressed, args.num_workers, args.start_at,
                                               args.write_database_ids)
    else:
        rewrite_single_cluster_from_npy(args.cluster_file, args.smiles_dir, args.output_dir,
                                        args.smiles_per_file, args.compressed, args.write_database_ids,
                                        args.overwrite)


if __name__ == '__main__':
    main()
