r"""Hidden module for HPC final round: merge all buffers and produce final clusters."""
import argparse
import time
import threading
from pathlib import Path
import pickle

from bblean import BitBirch
import psutil

from . import _config


class MemoryTracker:
    """Track peak RSS memory usage in a background thread."""
    def __init__(self):
        self.peak_memory_gb = 0.0
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._sample_memory, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def _sample_memory(self):
        proc = psutil.Process()
        while self.running:
            try:
                rss_gb = proc.memory_info().rss / 1024 / 1024 / 1024
                self.peak_memory_gb = max(self.peak_memory_gb, rss_gb)
            except Exception:
                pass
            time.sleep(0.1)

    def get_peak_memory_str(self):
        if self.peak_memory_gb > 0:
            return f" ({self.peak_memory_gb:.2f} GB peak)"
        return ""




def main(args: argparse.Namespace) -> None:
    """Process all buffer/index file pairs and produce final clusters.

    Merges all results from previous round into single tree and saves final output.
    """
    start_time = time.time()
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    try:
        output_dir = Path(args.output_dir)
        round_idx = args.round_idx
        file_pairs_str = args.file_pairs

        print(f"[Final Round] Starting final clustering")

        file_pairs = []
        for pair_str in file_pairs_str.split(","):
            buf_path, idx_path = pair_str.split(":")
            file_pairs.append((Path(buf_path), Path(idx_path)))

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

        total_time = time.time() - start_time
        mem_str = mem_tracker.get_peak_memory_str()
        print(f"[Final Round] ✓ Complete ({total_time:.2f}s{mem_str})")
        print(f"[Final Round] Results saved to: {output_dir}")
    finally:
        mem_tracker.stop()


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
