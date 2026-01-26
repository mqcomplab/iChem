from concurrent.futures import ProcessPoolExecutor
import logging
import argparse
from iChem.libchem import LibChem, LibComparison
import time
import psutil # type: ignore
import os

parser = argparse.ArgumentParser(description='Compare libraries in parallel')
parser.add_argument('-smi', '--smiles_files', type=str, required=True, nargs='+', help='Path(s) to library file(s)')
args = parser.parse_args()

# For as many files as provided, in a parallel fashion create a LibChem object for each and generate fingerprints
def create_libchem(smiles_file):
    process = psutil.Process(os.getpid())
    name = smiles_file.split('/')[-1].split('.')[0]
    
    logger = logging.getLogger(f'Process-{name}')
    
    def log_memory(stage):
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        logger.info(f'{stage}: RSS={mem_info.rss / 1024 / 1024:.2f}MB, VMS={mem_info.vms / 1024 / 1024:.2f}MB, {mem_percent:.2f}%')
    
    try:
        log_memory("START")
        
        lib = LibChem()
        log_memory("After LibChem init")
        
        lib.load_smiles(smiles_file)
        log_memory("After load_smiles")
        
        lib.generate_fingerprints(fp_type='ECFP4', n_bits=2048)
        log_memory("After generate_fingerprints")
        
        lib.cluster()
        log_memory("After cluster")
        
        lib.save_cluster_medoids()
        log_memory("After save_cluster_medoids")
        
        lib.empty_fps()
        log_memory("After empty_fps (FINAL)")
        
        return lib, name, process.memory_info().rss / 1024 / 1024
    
    except Exception as e:
        logger.error(f"ERROR in {name}: {str(e)}", exc_info=True)
        log_memory("ERROR STATE")
        raise

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('Main')
    smiles_files = args.smiles_files

    start = time.time()
    memory_stats = {}
    try:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(create_libchem, smiles_files))
        
        libchems = []
        for lib, name, peak_mem in results:
            libchems.append((lib, name))
            memory_stats[name] = peak_mem
            logger.info(f"Process {name} peak memory: {peak_mem:.2f}MB")
        
        logger.info(f"Total memory used across processes: {sum(memory_stats.values()):.2f}MB")
        logger.info(f"Average memory per process: {sum(memory_stats.values()) / len(memory_stats):.2f}MB")

        LibComp = LibComparison()
        for lib, name in libchems:
            LibComp.add_library(lib, name)
        counts, mapping = LibComp.cluster_classification_counts(verbose=True)

        total_time = time.time() - start
        print(f"Total parallel processing time: {total_time:.2f} seconds")
        print("Parallelized library comparison complete")

        LibComp.pie_chart_composition(save_path='library_composition_pie_chart.png')
        LibComp.plot_cluster_composition(lib_names=[name for _, name in libchems], top=25, save_path='library_cluster_composition.png')

        for key in mapping.keys():
            LibComp.cluster_visualization(cluster_number=mapping[key][0],
                                          save_path=f'cluster_{key}_structures.png')
            
        # For all the possible key combinations generate a pie chart
        from itertools import combinations
        names = [name for _, name in libchems]
        for r in range(2, len(names) + 1):
            for combo in combinations(names, r):
                LibComp.cluster_libraries(lib_names=list(combo))
                LibComp.pie_chart_composition(lib_names=list(combo),
                                                      save_path=f'cluster_combination_{"_".join(combo)}.png')
                
        # Save the n, iSIM, iSIM - sigma, n_clusters, and threshold of each library comparison to a .csv
        results = []
        for lib, name in libchems:
            iSIM = lib.get_iSIM()
            iSIM_sigma = lib.get_iSIM_sigma()
            n = lib.n_molecules
            n_clusters = len(lib.get_cluster_medoids(return_smiles=False))
            threshold = lib.threshold
            results.append((name, n, iSIM, iSIM_sigma, n_clusters, threshold))
        
        import pandas as pd # type: ignore
        df = pd.DataFrame(results, columns=['Library', 'N', 'iSIM', 'iSIM_sigma', 'N_clusters', 'Threshold'])
        df.to_csv('library_comparison_summary.csv', index=False)
        logger.info("Library comparison summary saved to library_comparison_summary.csv")

    except Exception as e:
        logger.error(f"Job failed with error: {str(e)}", exc_info=True)
        raise

