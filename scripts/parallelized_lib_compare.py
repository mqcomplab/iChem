from concurrent.futures import ProcessPoolExecutor
import logging
import argparse
from iChem.libchem import LibChem, LibComparison
import time
import psutil
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
        LibComp.cluster_classification_counts(verbose=True)
        
    except Exception as e:
        logger.error(f"Job failed with error: {str(e)}", exc_info=True)
        raise
    
    finally:
        total_time = time.time() - start
        print(f"Total parallel processing time: {total_time:.2f} seconds")
        print("Parallelized library comparison complete")



