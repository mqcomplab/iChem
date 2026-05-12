r"""Hidden module for HPC midsection round: merge and recluster batches."""
import argparse
import time
import multiprocessing as mp
import os
from pathlib import Path
import pickle

import numpy as np
from bblean import BitBirch
from bblean.fingerprints import _get_fps_file_num

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
    """Process batch of buffer/index file pairs, merge and recluster.

    Takes results from previous round and produces new buffers/indices for next round.
    """
    start_time = time.perf_counter()
    output_dir = Path(args.output_dir)
    round_idx = args.round_idx
    batch_label = args.batch_label
    file_pairs_str = args.file_pairs

    monitor_file = output_dir / f"memory-round{round_idx}-batch{batch_label}.csv"
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
        print(f"[Round {round_idx}, Batch {batch_label}] Starting midsection clustering")

        file_pairs = []
        for pair_str in file_pairs_str.split(","):
            buf_path, idx_path = pair_str.split(":")
            file_pairs.append((Path(buf_path).resolve(), Path(idx_path).resolve()))

        print(f"[Round {round_idx}, Batch {batch_label}] Processing {len(file_pairs)} file pairs")

        tree = BitBirch(
            threshold=args.threshold,
            branching_factor=args.branching_factor,
            merge_criterion=args.merge_criterion,
        )

        for buf_path, idx_path in file_pairs:
            print(f"[Round {round_idx}, Batch {batch_label}] Loading {buf_path.name}")
            with open(idx_path, "rb") as f:
                mol_idxs = pickle.load(f)
            tree._fit_buffers(buf_path, reinsert_index_seqs=mol_idxs)
            del mol_idxs

        if args.reclustering_iterations > 0:
            print(f"[Round {round_idx}, Batch {batch_label}] Reclustering ({args.reclustering_iterations} iterations)")
            tree.recluster_inplace(
                iterations=args.reclustering_iterations,
                extra_threshold=args.extra_threshold,
            )

        tree.delete_internal_nodes()

        print(f"[Round {round_idx}, Batch {batch_label}] Saving results")
        fps_bfs, mols_bfs = tree._bf_to_np()
        _save_bufs_and_mol_idxs(output_dir, fps_bfs, mols_bfs, batch_label, round_idx)

        # Cleanup only the specific files this job processed
        for buf_path, idx_path in file_pairs:
            try:
                buf_path.unlink()
            except Exception as e:
                print(f"[Round {round_idx}, Batch {batch_label}] Warning: Could not delete {buf_path.name}: {e}")
            try:
                idx_path.unlink()
            except Exception as e:
                print(f"[Round {round_idx}, Batch {batch_label}] Warning: Could not delete {idx_path.name}: {e}")

        total_time = time.perf_counter() - start_time
        peak_mem = _get_peak_memory_gib(output_dir)
        mem_str = f" ({peak_mem:.2f} GB peak)" if peak_mem > 0 else ""
        print(f"[Round {round_idx}, Batch {batch_label}] ✓ Complete ({total_time:.2f}s{mem_str})")
    finally:
        monitor_proc.terminate()
        monitor_proc.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HPC midsection round worker: merge and recluster batch of buffers"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--round-idx", type=int, required=True, help="Current round index")
    parser.add_argument("--batch-label", type=str, required=True, help="Batch label")
    parser.add_argument(
        "--file-pairs",
        type=str,
        required=True,
        help="Comma-separated file pairs (buf:idx,buf:idx,...)"
    )
    parser.add_argument("--threshold", type=float, required=True, help="BitBirch threshold")
    parser.add_argument("--branching-factor", type=int, required=True, help="BitBirch branching factor")
    parser.add_argument("--merge-criterion", type=str, default=_config.MERGE_CRITERION, help="Merge criterion")
    parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_MIDSECTION, help="Reclustering iterations")
    parser.add_argument("--extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold for reclustering")

    args = parser.parse_args()
    main(args)
