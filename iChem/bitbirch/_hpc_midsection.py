r"""Hidden module for HPC midsection round: merge and recluster batches."""
import argparse
import time
import threading
from pathlib import Path
import pickle

import numpy as np
from bblean import BitBirch
from bblean.fingerprints import _get_fps_file_num
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
    start_time = time.time()
    output_dir = Path(args.output_dir)
    round_idx = args.round_idx
    batch_label = args.batch_label
    file_pairs_str = args.file_pairs

    print(f"[Round {round_idx}, Batch {batch_label}] Starting midsection clustering")

    file_pairs = []
    for pair_str in file_pairs_str.split(","):
        buf_path, idx_path = pair_str.split(":")
        file_pairs.append((Path(buf_path), Path(idx_path)))

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

    total_time = time.time() - start_time
    print(f"[Round {round_idx}, Batch {batch_label}] ✓ Complete ({total_time:.2f}s)")


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
