from iChem.libchem.libchem_big import LibChemBig
import time
import tracemalloc
tracemalloc.start()

start = time.time()
Lib = LibChemBig(chunk_size=1_000_000, fp_type='ECFP4', n_bits=2048, library_name="TestLib")

# Use one line or the other, depending on if you need to genreate fingerprints

#Lib.load_fps_and_cluster("clustering_BRD4_train") # Directory containing the .npy files with the fingerprints
Lib.gen_fps_and_cluster("/home/kenneth/Documents/DELs/test_smi") # Directory containing the .smi


end = time.time()
print(f"Current memory usage: {tracemalloc.get_traced_memory()[0] / 1e9:.2f} GB")
print(f"Peak memory usage: {tracemalloc.get_traced_memory()[1] / 1e9:.2f} GB")
print(f"Total time taken: {end - start:.2f} seconds")