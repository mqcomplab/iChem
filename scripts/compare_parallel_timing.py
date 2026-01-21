import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from iChem.libchem import LibChem, LibComparison

# ============================================================================
# Non-parallelized version
# ============================================================================
def create_libchem_sequential(smiles_file):
    """Create a LibChem object for a single file"""
    lib = LibChem()
    lib.load_smiles(smiles_file)
    
    start_fp = time.time()
    lib.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
    fp_time = time.time() - start_fp
    
    start_cluster = time.time()
    lib.cluster()
    cluster_time = time.time() - start_cluster
    
    lib.save_cluster_medoids()
    lib.empty_fps()
    name = smiles_file.split('/')[-1].split('.')[0]
    return lib, name, fp_time, cluster_time


def run_sequential(smiles_files):
    """Run library comparison sequentially"""
    libchems = []
    total_fp_time = 0
    total_cluster_time = 0
    
    for smiles_file in smiles_files:
        lib, name, fp_time, cluster_time = create_libchem_sequential(smiles_file)
        libchems.append((lib, name))
        total_fp_time += fp_time
        total_cluster_time += cluster_time
    
    start_comp = time.time()
    LibComp = LibComparison()
    for lib, name in libchems:
        LibComp.add_library(lib, name)
    LibComp.cluster_classification_counts(verbose=True)
    comp_time = time.time() - start_comp
    
    return total_fp_time, total_cluster_time + comp_time


# ============================================================================
# Parallelized version
# ============================================================================
def create_libchem_parallel(smiles_file):
    """Create a LibChem object for a single file (for parallel execution)"""
    lib = LibChem()
    lib.load_smiles(smiles_file)
    
    lib.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
    
    lib.cluster()
    
    lib.save_cluster_medoids()
    lib.empty_fps()
    name = smiles_file.split('/')[-1].split('.')[0]
    return lib, name


def run_parallel(smiles_files):
    """Run library comparison in parallel"""
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(create_libchem_parallel, smiles_files))
    
    libchems = [(lib, name) for lib, name in results]

    LibComp = LibComparison()
    for lib, name in libchems:
        LibComp.add_library(lib, name)
    LibComp.cluster_classification_counts(verbose=True)

# ============================================================================
# Main timing comparison
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare parallelized vs sequential library comparison')
    parser.add_argument('-smi', '--smiles_files', type=str, required=True, nargs='+', help='Path(s) to library file(s)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    smiles_files = args.smiles_files

    print("\n" + "="*70)
    print("LIBRARY COMPARISON: PARALLEL vs SEQUENTIAL TIMING")
    print("="*70)

    # Run sequential version
    print("Running SEQUENTIAL version...")
    start_time = time.time()
    seq_fp_time, seq_cluster_time = run_sequential(smiles_files)
    sequential_time = time.time() - start_time

    # Run parallel version
    print("Running PARALLEL version...")
    start_time = time.time()
    run_parallel(smiles_files)
    parallel_time = time.time() - start_time

    # Print results
    print("\n" + "="*70)
    print("TIMING RESULTS")
    print("="*70)
    print(f"{'Method':<30} {'Total Time':<20}")
    print("-"*70)
    print(f"{'Sequential (FP + Clustering)':<30} {seq_fp_time:>6.2f}s + {seq_cluster_time:>6.2f}s = {sequential_time:>6.2f}s")
    print(f"{'Parallel':<30} {parallel_time:>6.2f}s")
    print("="*70 + "\n")
