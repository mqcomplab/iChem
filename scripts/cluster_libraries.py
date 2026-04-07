import argparse
import numpy as np # type: ignore
import pickle as pkl
import time

from iChem.bblean import BitBirch

# Script to cluster the BFs from multiple libraries and then get the memberships per cluster from each library
parser = argparse.ArgumentParser(description='Compare libraries in parallel')
parser.add_argument('-bfs',
                    '--bfs_files',
                    type=str,
                    required=True,
                    nargs='+',
                    help='Path(s) to the BFs file(s)')
parser.add_argument('-t',
                    '--threshold',
                    type=float,
                    help='Clustering threshold',
                    required=True)
args = parser.parse_args()

start = time.time()

# Sort the input files to ensure consistent ordering
bfs_files = sorted(args.bfs_files)

# Create a new BitBirch model to fit all the BFs in the final clustering step
bb_object = BitBirch(threshold=args.threshold,
                     branching_factor=1024,
                     merge_criterion='diameter')

# Read the BFs
for k, bf_file in enumerate(bfs_files):
    # Load the BFs and molecule IDs from the file
    with open(bf_file, 'rb') as f:
        bfs, mol_ids = pkl.load(f)
    print(f"Loaded {len(bfs)} BFs from {bf_file}")

    # Fit the BFs into the BitBirch model
    for key in bfs.keys():
        bf_to_fit = bfs[key]
        mols_to_fit = mol_ids[key]
        bb_object._fit_np(X=bf_to_fit, reinsert_index_seqs=mols_to_fit, db_name=chr(ord('A') + k))

# Get a refinement of the clustering to reduce the number of singletons
bb_object.recluster_inplace(iterations=3, extra_threshold=0.025)

# Dump the final BFs for analysis
final_bfs, final_mol_ids = bb_object._bf_to_np()
final_output_path = "final_cluster_mixed_libs.pkl"
with open(final_output_path, 'wb') as f:
    pkl.dump((final_bfs, final_mol_ids), f)

print(f"Final clustering time: {time.time() - start:.2f} seconds")
print(f"Final cluster ids saved to {final_output_path}")
print(f"Final number of clusters: {np.sum([len(final_bfs[key]) for key in final_bfs.keys()])}")
print(f"Number of singletons in final clusters: {np.sum([np.sum([len(mol_ids_cluster) == 1 for mol_ids_cluster in final_mol_ids[key]]) for key in final_mol_ids.keys()])}")