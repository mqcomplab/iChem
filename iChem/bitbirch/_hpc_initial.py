r"""Hidden module for HPC initial round: fingerprint generation + clustering."""
import argparse
import time
import threading
from pathlib import Path
import gzip as gz
import pickle

import numpy as np
from bblean import BitBirch
from bblean.fingerprints import _get_fps_file_num
import psutil

from ..utils import binary_fps, load_smiles
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
    """Load SMILES files, generate fingerprints, perform initial clustering.

    Uses global molecule indices to preserve molecule identity across batches.
    Saves results to the specified output directory.
    """
    start_time = time.time()
    mem_tracker = MemoryTracker()
    mem_tracker.start()

    try:
        smi_files = [Path(f) for f in args.smi_files]
        start_idx = args.start_idx
        end_idx = args.end_idx
        label = args.label
        output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd().resolve()

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{label}] Processing {len(smi_files)} SMILES files (global indices {start_idx}-{end_idx})")
        print(f"[{label}] Results directory: {output_dir}")

        all_smiles = []
        for smi_file in smi_files:
            print(f"[{label}] Loading {smi_file}")
            is_smi_gz = smi_file.name.endswith('.smi.gz')

            if smi_file.suffix == '.smi':
                smiles = load_smiles(smi_file)
            elif is_smi_gz:
                with gz.open(smi_file, 'rt') as f:
                    smiles = load_smiles(f)
            else:
                raise ValueError(f"Unsupported file type: {smi_file.suffix}")

            all_smiles.extend(smiles)

        print(f"[{label}] Total molecules loaded: {len(all_smiles)}")
        expected_count = end_idx - start_idx
        if len(all_smiles) != expected_count:
            print(f"[{label}] Warning: Expected {expected_count} molecules, got {len(all_smiles)}")

        print(f"[{label}] Generating {args.fp_type} fingerprints ({args.n_bits} bits)")
        fps, invalid_ids = binary_fps(
            all_smiles,
            fp_type=args.fp_type,
            n_bits=args.n_bits,
            packed=True,
            return_invalid=True,
        )

        if invalid_ids:
            print(f"[{label}] Removed {len(invalid_ids)} invalid molecules")
            idx_range = [start_idx + i for i in range(len(all_smiles)) if i not in invalid_ids]
        else:
            idx_range = list(range(start_idx, end_idx))

        npy_path = output_dir / f"temp_fps_{label}.npy"
        np.save(npy_path, fps)
        print(f"[{label}] Saved fingerprints to {npy_path}")

        del fps
        del all_smiles

        if len(idx_range) != _get_fps_file_num(npy_path):
            raise ValueError(
                f"Mismatch: {len(idx_range)} indices but {_get_fps_file_num(npy_path)} fps"
            )

        print(f"[{label}] Starting BitBirch clustering")
        tree = BitBirch(
            threshold=args.threshold,
            branching_factor=args.branching_factor,
            merge_criterion=args.merge_criterion,
        )
        tree.fit(npy_path, reinsert_indices=idx_range)

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

        total_time = time.time() - start_time
        mem_str = mem_tracker.get_peak_memory_str()
        print(f"[{label}] ✓ Complete ({total_time:.2f}s{mem_str})")
    finally:
        mem_tracker.stop()


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
