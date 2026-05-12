r"""Hidden module for HPC initial round: fingerprint generation + clustering."""
import argparse
import time
import multiprocessing as mp
import os
from pathlib import Path
import gzip as gz
import pickle

import numpy as np
from bblean import BitBirch
from bblean.fingerprints import _get_fps_file_num

from ..utils import binary_fps, load_smiles
from . import _config

_BYTES_TO_GIB = 1 / 1024**3


def _monitor_rss_process(file: Path | str, interval_s: float, start_time: float, parent_pid: int) -> None:
    """Monitor RSS peak (parent + all children)."""
    import psutil
    file = Path(file)
    this_pid = os.getpid()
    ps = psutil.Process(parent_pid)

    def total_rss() -> float:
        total_rss = ps.memory_info().rss
        for proc in ps.children(recursive=True):
            if proc.pid == this_pid:
                continue
            try:
                total_rss += proc.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return total_rss

    with open(file, mode="w", encoding="utf-8") as f:
        f.write("rss_gib,time_s\n")
        f.flush()
        os.fsync(f.fileno())

    max_rss_gib = 0.0
    while True:
        total_rss_gib = total_rss() * _BYTES_TO_GIB
        with open(file, mode="a", encoding="utf-8") as f:
            f.write(f"{total_rss_gib},{time.perf_counter() - start_time}\n")
            f.flush()
            os.fsync(f.fileno())
        if total_rss_gib > max_rss_gib:
            max_rss_gib = total_rss_gib
            with open(file.parent / "max-rss.json", mode="w", encoding="utf-8") as f:
                f.write(f"{max_rss_gib}\n")
                f.flush()
                os.fsync(f.fileno())
        time.sleep(interval_s)


def _get_peak_memory_gib(out_dir: Path) -> float:
    """Read peak memory from monitor daemon."""
    file = out_dir / "max-rss.json"
    try:
        if file.exists():
            with open(file, mode="r", encoding="utf-8") as f:
                return float(f.read().strip())
    except Exception:
        pass
    return 0.0





def _numpy_streaming_save(fp_list, path: Path | str) -> None:
    """Save a list of numpy arrays efficiently in streaming fashion."""
    first_arr = np.ascontiguousarray(fp_list[0])
    header = np.lib.format.header_data_from_array_1_0(first_arr)
    header["shape"] = (len(fp_list), len(first_arr))
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npy")
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, header)
        for arr in fp_list:
            np.ascontiguousarray(arr).tofile(f)


def _save_bufs_and_mol_idxs(
    out_dir: Path,
    fps_bfs: dict[str, any],
    mols_bfs: dict[str, any],
    label: str,
    round_idx: int,
) -> None:
    """Save fingerprint buffers and molecule indices to disk."""
    for dtype, buf_list in fps_bfs.items():
        suffix = f".label-{label}-{dtype.replace('8', '08')}"
        _numpy_streaming_save(buf_list, out_dir / f"round-{round_idx}-bufs{suffix}.npy")
        with open(out_dir / f"round-{round_idx}-idxs{suffix}.pkl", mode="wb") as f:
            pickle.dump(mols_bfs[dtype], f)


def main(args: argparse.Namespace) -> None:
    """Load SMILES files, generate fingerprints, perform initial clustering.

    Uses global molecule indices to preserve molecule identity across batches.
    Saves results to the specified output directory.
    """
    start_time = time.perf_counter()
    smi_files = [Path(f) for f in args.smi_files]
    start_idx = args.start_idx
    end_idx = args.end_idx
    label = args.label
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    monitor_file = output_dir / f"memory-{label}.csv"
    monitor_proc = mp.Process(
        target=_monitor_rss_process,
        kwargs=dict(
            file=monitor_file,
            interval_s=0.1,
            start_time=start_time,
            parent_pid=os.getpid(),
        ),
        daemon=True,
    )
    monitor_proc.start()

    try:
        print(f"[{label}] Processing {len(smi_files)} SMILES files (global indices {start_idx}-{end_idx})")
        print(f"[{label}] Results directory: {output_dir}")

        print(f"[{label}] Starting BitBirch clustering")
        tree = BitBirch(
            threshold=args.threshold,
            branching_factor=args.branching_factor,
            merge_criterion=args.merge_criterion,
        )

        current_idx = start_idx
        for smi_file in smi_files:
            print(f"[{label}] Loading {smi_file}")
            is_smi_gz = smi_file.name.endswith('.smi.gz')

            if smi_file.suffix == '.smi':
                smiles = load_smiles(smi_file)
            elif is_smi_gz:
                smiles = []
                with gz.open(smi_file, 'rt') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            smiles.append(line.split()[0])
            else:
                raise ValueError(f"Unsupported file type: {smi_file.suffix}")

            print(f"[{label}] Loaded {len(smiles)} SMILES from {smi_file}")
            idx_range = range(current_idx, current_idx + len(smiles))
            current_idx += len(smiles)
            print(f"[{label}] Assigning global indices {idx_range.start} to {idx_range.stop - 1}")

            print(f"[{label}] Generating {args.fp_type} fingerprints ({args.n_bits} bits)")
            fps, invalid_ids = binary_fps(
                smiles,
                fp_type=args.fp_type,
                n_bits=args.n_bits,
                packed=True,
                return_invalid=True,
            )

            if invalid_ids:
                print(f"[{label}] Warning: Found {len(invalid_ids)} invalid SMILES in {smi_file}")
                valid_indices = [idx for i, idx in enumerate(idx_range) if i not in invalid_ids]
                print(f"[{label}] Retaining {len(valid_indices)} valid SMILES")
            else:
                valid_indices = list(idx_range)

            del smiles

            npy_path = output_dir / f"temp_fps_{label}.npy"
            np.save(npy_path, fps)
            print(f"[{label}] Saved temporary fingerprints to {npy_path}")
            del fps

            if len(valid_indices) != _get_fps_file_num(npy_path):
                raise ValueError(
                    f"Mismatch: {len(valid_indices)} indices but {_get_fps_file_num(npy_path)} fps"
                )

            tree.fit(npy_path, reinsert_indices=valid_indices)

        if args.reclustering_iterations > 0:
            print(f"[{label}] Reclustering ({args.reclustering_iterations} iterations)")
            tree.recluster_inplace(
                iterations=args.reclustering_iterations,
                extra_threshold=args.extra_threshold,
            )

        tree.delete_internal_nodes()

        print(f"[{label}] Saving results")
        fps_bfs, mols_bfs = tree._bf_to_np()
        _save_bufs_and_mol_idxs(output_dir, fps_bfs, mols_bfs, label, 1)

        try:
            npy_path.unlink()
            print(f"[{label}] Cleaned up temporary file")
        except Exception as e:
            print(f"[{label}] Warning: Could not delete {npy_path}: {e}")

        total_time = time.perf_counter() - start_time
        peak_mem = _get_peak_memory_gib(output_dir)
        mem_str = f" ({peak_mem:.2f} GB peak)" if peak_mem > 0 else ""
        print(f"[{label}] ✓ Complete ({total_time:.2f}s{mem_str})")
    finally:
        monitor_proc.terminate()
        monitor_proc.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HPC initial round worker: clustering individual SMILES batches"
    )
    parser.add_argument("--smi-files", nargs="+", required=True, help="Input SMILES files")
    parser.add_argument("--start-idx", type=int, required=True, help="Global starting index")
    parser.add_argument("--end-idx", type=int, required=True, help="Global ending index")
    parser.add_argument("--label", type=str, required=True, help="Batch label")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base directory for results (default: current directory)"
    )
    parser.add_argument("--threshold", type=float, required=True, help="BitBirch threshold")
    parser.add_argument("--branching-factor", type=int, required=True, help="BitBirch branching factor")
    parser.add_argument("--merge-criterion", type=str, default=_config.MERGE_CRITERION, help="Merge criterion")
    parser.add_argument("--fp-type", type=str, default=_config.FINGERPRINT_TYPE, help="Fingerprint type")
    parser.add_argument("--n-bits", type=int, default=_config.N_BITS, help="Number of bits")
    parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_INITIAL, help="Reclustering iterations")
    parser.add_argument("--extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold for reclustering")

    args = parser.parse_args()
    main(args)
