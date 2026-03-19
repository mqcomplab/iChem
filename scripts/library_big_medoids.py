from iChem.libchem.libchem_big import LibChemBig
import argparse
import time

if __name__ == "__main__":
    
    start = time.time()
    parser = argparse.ArgumentParser(description="Cluster a large chemical library and save medoids.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .smi files with the SMILES.")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="Number of molecules per chunk.")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of parallel workers (default: all cores).")
    parser.add_argument("--fp_type", type=str, default='ECFP4', help="Type of fingerprint (default: ECFP4).")
    parser.add_argument("--n_bits", type=int, default=2048, help="Number of bits for fingerprints (default: 2048).")
    parser.add_argument("--threshold", type=float, default=None, help="Clustering threshold (default: optimal).")
    parser.add_argument("--library_name", type=str, default="Lib1Big", help="Name for the library (used in output filenames).")
    
    args = parser.parse_args()
    
    lib = LibChemBig(
        chunk_size=args.chunk_size,
        n_workers=args.n_workers,
        fp_type=args.fp_type,
        n_bits=args.n_bits,
        threshold=args.threshold,
        library_name=args.library_name
    )

    lib.gen_fps(args.input_dir)

    print("Finished generating fingerprints. Starting clustering...")
    print(time.time() - start, "seconds elapsed for fingerprint generation.")
    
    lib.cluster()

    print("Finished clustering. Saving medoids...")
    print(time.time() - start, "seconds elapsed for clustering.")

    lib.save_cluster_medoids()

    print("Medoids saved. Final statistics:")
    print(time.time() - start, "seconds elapsed for entire process.")

    print(lib.dump_statistics())

    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")