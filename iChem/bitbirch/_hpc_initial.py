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
from .zinc22_ids import pack_zinc22_uint64

_BYTES_TO_GIB = 1 / 1024**3


def _monitor_rss_process(max_rss_value, parent_pid: int, interval_s: float) -> None:
    """Monitor RSS peak for parent process only."""
    import psutil
    ps = psutil.Process(parent_pid)

    while True:
        try:
            rss_gib = ps.memory_info().rss * _BYTES_TO_GIB
            if rss_gib > max_rss_value.value:
                max_rss_value.value = rss_gib
        except psutil.NoSuchProcess:
            break
        except Exception:
            pass
        time.sleep(interval_s)








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


def _load_smiles_and_packed_zinc22_ids(smi_file: Path) -> tuple[list[str], list[int], int]:
    """Load SMILES/ZINC_ID records, excluding malformed ID records.

    Comma-, tab-, and whitespace-separated two-column records are supported.
    The returned IDs are aligned with the returned SMILES; fingerprint errors
    are filtered later using that same alignment.
    """
    opener = gz.open if smi_file.name.endswith(".smi.gz") else open
    smiles: list[str] = []
    packed_ids: list[int] = []
    invalid_records = 0

    with opener(smi_file, "rt") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            record = line.strip()
            if not record:
                continue
            try:
                if "," in record:
                    smiles_value, zinc_id = record.split(",", 1)
                else:
                    smiles_value, zinc_id = record.split(None, 1)
                smiles_value = smiles_value.strip()
                if not smiles_value:
                    raise ValueError("empty SMILES")
                packed_id = pack_zinc22_uint64(zinc_id.strip())
            except ValueError as error:
                invalid_records += 1
                print(
                    f"Skipping malformed SMILES/ZINC_ID record at "
                    f"{smi_file}:{line_number}: {error}"
                )
                continue
            smiles.append(smiles_value)
            packed_ids.append(packed_id)

    return smiles, packed_ids, invalid_records


def main(args: argparse.Namespace) -> None:
    """Load SMILES files, generate fingerprints, perform initial clustering.

    Uses global molecule indices by default, or packed ZINC22 IDs when
    ``--code-ids`` is supplied, to preserve molecule identity across batches.
    Saves results to the specified output directory.
    """
    start_time = time.perf_counter()
    smi_files = [Path(f) for f in args.smi_files]
    start_idx = args.start_idx
    end_idx = args.end_idx
    label = args.label
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    max_rss_value = mp.Value('d', 0.0)
    monitor_proc = mp.Process(
        target=_monitor_rss_process,
        args=(max_rss_value, os.getpid(), 0.1),
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

            packed_zinc_ids: list[int] | None = None
            if args.code_ids:
                smiles, packed_zinc_ids, invalid_records = _load_smiles_and_packed_zinc22_ids(smi_file)
                if invalid_records:
                    print(f"[{label}] Skipped {invalid_records} malformed SMILES/ZINC_ID records")
            elif smi_file.suffix == '.smi':
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
            if args.code_ids:
                print(f"[{label}] Using packed ZINC22 IDs as molecule indices")
            else:
                print(f"[{label}] Assigning global indices {idx_range.start} to {idx_range.stop - 1}")

            if not smiles:
                print(f"[{label}] Warning: No valid input records in {smi_file}; skipping")
                continue

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
                source_indices = packed_zinc_ids if args.code_ids else idx_range
                valid_indices = [idx for i, idx in enumerate(source_indices) if i not in invalid_ids]
                print(f"[{label}] Retaining {len(valid_indices)} valid SMILES")
            else:
                valid_indices = list(packed_zinc_ids if args.code_ids else idx_range)

            del smiles

            if not valid_indices:
                print(f"[{label}] Warning: No valid fingerprints in {smi_file}; skipping")
                del fps
                continue

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
        mem_str = f" ({max_rss_value.value:.2f} GB peak)" if max_rss_value.value > 0 else ""
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
    parser.add_argument(
        "--code-ids",
        action="store_true",
        default=False,
        help="Read SMILES and ZINC_ID records (comma, tab, or space separated) as packed molecule indices",
    )

    args = parser.parse_args()
    main(args)
