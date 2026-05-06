"""Command line entry points for iChem."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Sequence

from .bitbirch.cluster import cluster
from .bitbirch.multiround_reclustering import run_multiround_reclustering


def _print_banner() -> None:
    """Print iChem banner with group attribution."""
    banner = r"""
      _     _     _     _     _
     / \   / \   / \   / \   / \
    |010|-|101|-|010|-|101|-|010|
     \_/   \_/   \_/   \_/   \_/
    
    iChem: Instant Cheminformatics
      _     _     _     _     _    
     / \   / \   / \   / \   / \
    |101|-|010|-|101|-|010|-|101|
     \_/   \_/   \_/   \_/   \_/

    Miranda-Quintana Group
    University of Florida
    Department of Chemistry
    """
    print(banner)


def _path_list(value: str) -> list[Path]:
    path = Path(value)
    if path.is_dir():
        return sorted(path.glob("*.npy"))
    return [path]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iChem", description="iChem BitBIRCH tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cluster_parser = subparsers.add_parser("cluster", help="Cluster a file or directory")
    cluster_parser.add_argument("input", type=Path, help="Path to a .smi, .smi.gz, .npy file, or a directory")
    cluster_parser.add_argument("--threshold", type=float, default=None)
    cluster_parser.add_argument("--fp-type", default="ECFP4")
    cluster_parser.add_argument("--n-bits", type=int, default=2048)
    cluster_parser.add_argument("--branching-factor", type=int, default=1024)
    cluster_parser.add_argument("--merge-criterion", default="diameter")
    cluster_parser.add_argument("--out", type=Path, default=None, help="Optional pickle output path")
    cluster_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    multiround_parser = subparsers.add_parser("multiround", help="Run multi-round clustering")
    multiround_parser.add_argument("input", nargs="+", type=Path, help="Input .npy files or a directory containing them")
    multiround_parser.add_argument("--out-dir", type=Path, required=True)
    multiround_parser.add_argument("--n-features", type=int, default=None)
    multiround_parser.add_argument("--input-is-packed", action=argparse.BooleanOptionalAction, default=True)
    multiround_parser.add_argument("--num-initial-processes", type=int, default=10)
    multiround_parser.add_argument("--num-midsection-processes", type=int, default=None)
    multiround_parser.add_argument("--merge-criterion", default="diameter")
    multiround_parser.add_argument("--branching-factor", type=int, default=1024)
    multiround_parser.add_argument("--threshold", type=float, default=0.65)
    multiround_parser.add_argument("--midsection-threshold-change", type=float, default=0.0)
    multiround_parser.add_argument("--num-midsection-rounds", type=int, default=1)
    multiround_parser.add_argument("--bin-size", type=int, default=10)
    multiround_parser.add_argument("--max-tasks-per-process", type=int, default=1)
    multiround_parser.add_argument("--save-tree", action=argparse.BooleanOptionalAction, default=False)
    multiround_parser.add_argument("--save-centroids", action=argparse.BooleanOptionalAction, default=True)
    multiround_parser.add_argument("--reclustering-iterations-initial", type=int, default=3)
    multiround_parser.add_argument("--reclustering-iterations-midsection", type=int, default=0)
    multiround_parser.add_argument("--reclustering-iterations-final", type=int, default=0)
    multiround_parser.add_argument("--reclustering-extra-threshold", type=float, default=0.025)
    multiround_parser.add_argument("--max-fps", type=int, default=None)
    multiround_parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    multiround_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    return parser


def _run_cluster(args: argparse.Namespace) -> int:
    result = cluster(
        args.input,
        threshold=args.threshold,
        fp_type=args.fp_type,
        n_bits=args.n_bits,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        verbose=args.verbose,
    )
    if args.out is None:
        output_path = Path('cluster_output.pkl')
    else:
        output_path = args.out

    with open(output_path, "wb") as handle:
        pickle.dump(result, handle)
    print(f"Saved clustering output to {output_path}")
    return 0


def _run_multiround(args: argparse.Namespace) -> int:
    input_files: list[Path] = []
    for value in args.input:
        input_files.extend(_path_list(str(value)))
    if not input_files:
        raise ValueError("No input files were found")

    timer = run_multiround_reclustering(
        input_files=input_files,
        out_dir=args.out_dir,
        n_features=args.n_features,
        input_is_packed=args.input_is_packed,
        num_initial_processes=args.num_initial_processes,
        num_midsection_processes=args.num_midsection_processes,
        merge_criterion=args.merge_criterion,
        branching_factor=args.branching_factor,
        threshold=args.threshold,
        midsection_threshold_change=args.midsection_threshold_change,
        num_midsection_rounds=args.num_midsection_rounds,
        bin_size=args.bin_size,
        max_tasks_per_process=args.max_tasks_per_process,
        save_tree=args.save_tree,
        save_centroids=args.save_centroids,
        reclustering_iterations_initial=args.reclustering_iterations_initial,
        reclustering_iterations_midsection=args.reclustering_iterations_midsection,
        reclustering_iterations_final=args.reclustering_iterations_final,
        reclustering_extra_threshold=args.reclustering_extra_threshold,
        max_fps=args.max_fps,
        verbose=args.verbose,
        cleanup=args.cleanup,
    )
    print(timer)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    
    _print_banner()
    
    if args.command == "cluster":
        return _run_cluster(args)
    if args.command == "multiround":
        return _run_multiround(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
