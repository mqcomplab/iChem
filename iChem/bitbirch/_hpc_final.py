r"""Hidden module for HPC final round: merge all buffers and produce final clusters."""
import argparse
import time
import multiprocessing as mp
import os
from pathlib import Path
import pickle

from bblean import BitBirch

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




def main(args: argparse.Namespace) -> None:
    """Process all buffer/index file pairs and produce final clusters.

    Merges all results from previous round into single tree and saves final output.
    """
    start_time = time.perf_counter()
    output_dir = Path(args.output_dir)
    round_idx = args.round_idx
    file_pairs_str = args.file_pairs

    monitor_file = output_dir / "memory-final.csv"
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
        print(f"[Final Round] Starting final clustering")

        file_pairs = []
        for pair_str in file_pairs_str.split(","):
            buf_path, idx_path = pair_str.split(":")
            file_pairs.append((Path(buf_path).resolve(), Path(idx_path).resolve()))

        print(f"[Final Round] Processing {len(file_pairs)} file pairs")

        tree = BitBirch(
            threshold=args.threshold,
            branching_factor=args.branching_factor,
            merge_criterion=args.merge_criterion,
        )

        for buf_path, idx_path in file_pairs:
            print(f"[Final Round] Loading {buf_path.name}")
            with open(idx_path, "rb") as f:
                mol_idxs = pickle.load(f)
            tree._fit_buffers(buf_path, reinsert_index_seqs=mol_idxs)
            del mol_idxs

        if args.reclustering_iterations > 0:
            print(f"[Final Round] Reclustering ({args.reclustering_iterations} iterations)")
            tree.recluster_inplace(
                iterations=args.reclustering_iterations,
                extra_threshold=args.extra_threshold,
            )

        print(f"[Final Round] Saving final results")

        if args.save_tree:
            tree.save(output_dir / "bitbirch.pkl")
            print(f"[Final Round] Saved tree to {output_dir / 'bitbirch.pkl'}")

        tree.delete_internal_nodes()

        if args.save_centroids:
            output = tree.get_centroids_mol_ids()
            with open(output_dir / "clusters.pkl", mode="wb") as f:
                pickle.dump(output["mol_ids"], f)
            with open(output_dir / "cluster-centroids-packed.pkl", mode="wb") as f:
                pickle.dump(output["centroids"], f)
            print(f"[Final Round] Saved centroids and cluster assignments")
        else:
            with open(output_dir / "clusters.pkl", mode="wb") as f:
                pickle.dump(tree.get_cluster_mol_ids(), f)
            print(f"[Final Round] Saved cluster assignments")

        # Cleanup all round files
        for buf_file in output_dir.glob("round-*-bufs*.npy"):
            try:
                buf_file.unlink()
            except Exception as e:
                print(f"[Final Round] Warning: Could not delete {buf_file.name}: {e}")
        for idx_file in output_dir.glob("round-*-idxs*.pkl"):
            try:
                idx_file.unlink()
            except Exception as e:
                print(f"[Final Round] Warning: Could not delete {idx_file.name}: {e}")

        total_time = time.perf_counter() - start_time
        peak_mem = _get_peak_memory_gib(output_dir)
        mem_str = f" ({peak_mem:.2f} GB peak)" if peak_mem > 0 else ""
        print(f"[Final Round] ✓ Complete ({total_time:.2f}s{mem_str})")
        print(f"[Final Round] Results saved to: {output_dir}")
    finally:
        monitor_proc.terminate()
        monitor_proc.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HPC final round worker: merge all buffers and produce final clusters"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--round-idx", type=int, required=True, help="Previous round index to read from")
    parser.add_argument(
        "--file-pairs",
        type=str,
        required=True,
        help="Comma-separated file pairs (buf:idx,buf:idx,...)"
    )
    parser.add_argument("--threshold", type=float, required=True, help="BitBirch threshold")
    parser.add_argument("--branching-factor", type=int, required=True, help="BitBirch branching factor")
    parser.add_argument("--merge-criterion", type=str, default=_config.MERGE_CRITERION, help="Merge criterion")
    parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_FINAL, help="Reclustering iterations")
    parser.add_argument("--extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold for reclustering")
    parser.add_argument("--save-tree", action="store_true", default=False, help="Save the BitBirch tree")
    parser.add_argument("--save-centroids", action="store_true", default=True, help="Save centroids and cluster assignments")

    args = parser.parse_args()
    main(args)
