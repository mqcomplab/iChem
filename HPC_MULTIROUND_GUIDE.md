# HPC MultiRound Reclustering - Complete Guide

Distributed SLURM-based BitBirch clustering for massive molecular libraries.

## Quick Start

```bash
# 1. Generate initial round jobs
iChem initial-round *.smi \
  --out-dir ./clustering \
  --files-per-job 10 \
  --verbose

# 2. Submit initial jobs
./clustering/submit_initial_jobs.sh

# 3. Wait for completion, then generate midsection jobs
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 2 \
  --bin-size 5 \
  --verbose

# 4. Submit midsection jobs
./clustering/submit_midsection_round_2_jobs.sh

# 5. Generate final job
iChem final-round \
  --output-dir ./clustering \
  --prev-round-idx 2 \
  --verbose

# 6. Submit final job
./clustering/submit_final_round_job.sh

# Results: clusters.pkl, cluster-centroids-packed.pkl
```

---

## Architecture Overview

The refactored HPC pipeline splits clustering into independent phases, each with its own CLI command and worker modules:

### Phase 1: Initial Round (Fingerprint + Clustering)
- **CLI Command**: `iChem initial-round`
- **Generator Module**: `_hpc_initial_submit.py`
- **Worker Module**: `_hpc_initial.py`
- **Output**: `submit_initial_jobs.sh`
- **Each Job**:
  - Loads SMILES from file chunk
  - Generates fingerprints (ECFP4, configurable)
  - Performs BitBirch clustering
  - Saves buffers and indices: `round-1-bufs*.npy`, `round-1-idxs*.pkl`
- **Memory**: 16GB, 1 CPU per job
- **Runs**: N jobs in parallel (one per file batch)

### Phase 2: Midsection Rounds (Merge & Cluster)
- **CLI Command**: `iChem midsection-round`
- **Generator Module**: `_hpc_midsection_submit.py`
- **Worker Module**: `_hpc_midsection.py`
- **Output**: `submit_midsection_round_{N}_jobs.sh`
- **Each Job**:
  - Loads buffers from previous round file pairs
  - Merges BitFeatures from multiple batches
  - Performs reclustering (optional)
  - Saves new buffers: `round-{N}-bufs*.npy`, `round-{N}-idxs*.pkl`
- **Memory**: 48GB, 3 CPUs per job
- **Runs**: M jobs in parallel (one per batch, batch size = `--bin-size`)
- **Repeatable**: Chain multiple midsection rounds

### Phase 3: Final Round (Consolidation)
- **CLI Command**: `iChem final-round`
- **Generator Module**: `_hpc_final_submit.py`
- **Worker Module**: `_hpc_final.py`
- **Output**: `submit_final_round_job.sh`
- **Single Job**:
  - Loads ALL buffers from previous round
  - Performs final tree consolidation
  - Saves output:
    - `clusters.pkl` - Molecule → Cluster ID mapping
    - `cluster-centroids-packed.pkl` - Cluster centroids (optional)
    - `bitbirch.pkl` - Full tree object (optional)
- **Memory**: 96GB, 6 CPUs
- **Runs**: 1 job

---

## Configuration

All defaults centralized in `iChem/bitbirch/_config.py`. Override via CLI flags.

### Global Defaults
```python
THRESHOLD = 0.3                              # BitBirch threshold
BRANCHING_FACTOR = 10_000                    # BitBirch branching factor
MERGE_CRITERION = "diameter"                 # "diameter" or "radius"
FINGERPRINT_TYPE = "ECFP4"                   # Fingerprint type RDKIT, MACCS, AP
N_BITS = 2048                                # Fingerprint bits
```

### Reclustering
```python
RECLUSTERING_ITERATIONS_INITIAL = 3
RECLUSTERING_ITERATIONS_MIDSECTION = 0
RECLUSTERING_ITERATIONS_FINAL = 0
RECLUSTERING_EXTRA_THRESHOLD = 0.025
```

### SLURM Resources
```python
SLURM_MEM_INITIAL = "16G"                    # Initial round memory
SLURM_CPUS_INITIAL = 1
SLURM_MEM_MIDSECTION = "48G"                 # Midsection memory
SLURM_CPUS_MIDSECTION = 3
SLURM_MEM_FINAL = "96G"                      # Final round memory
SLURM_CPUS_FINAL = 6
SLURM_TIME = "24:00:00"                      # All jobs
SLURM_PARTITION = ""                         # Empty = default partition (CPU)
```

**Note**: All jobs are CPU-only. Leave `--slurm-partition` empty (default) or specify your CPU partition.

### HPC Parameters
```python
FILES_PER_JOB = 10                           # Files per initial job
BIN_SIZE = 5                                 # Buffer pairs per midsection job
```

---

## CLI Commands

### `iChem initial-round`

Generate and view initial round job submission script.

```bash
iChem initial-round \
  FILE1.smi FILE2.smi FILE3.smi \
  --out-dir ./clustering \
  --files-per-job 10 \
  --threshold 0.3 \
  --branching-factor 10000 \
  --merge-criterion diameter \
  --fp-type ECFP4 \
  --n-bits 2048 \
  --reclustering-iterations 3 \
  --reclustering-extra-threshold 0.025 \
  --slurm-mem 16G \
  --slurm-cpus 1 \
  --slurm-time 24:00:00 \
  --result-base-dir ./results \
  --verbose
```

**Output**: `./clustering/submit_initial_jobs.sh`
- Contains multiple `sbatch` commands (one per batch)
- User runs: `./clustering/submit_initial_jobs.sh`

### `iChem midsection-round`

Generate midsection round job submission script.

```bash
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 2 \
  --bin-size 5 \
  --threshold 0.3 \
  --branching-factor 10000 \
  --merge-criterion diameter \
  --reclustering-iterations 0 \
  --reclustering-extra-threshold 0.025 \
  --slurm-mem 48G \
  --slurm-cpus 3 \
  --slurm-time 24:00:00 \
  --verbose
```

**Automatically Discovers**:
- Previous round files: `round-{round_idx-1}-bufs*.npy`, `round-{round_idx-1}-idxs*.pkl`
- Chunks them by `--bin-size`
- Creates one job per chunk

**Output**: `./clustering/submit_midsection_round_2_jobs.sh`

### `iChem final-round`

Generate final round job submission script.

```bash
iChem final-round \
  --output-dir ./clustering \
  --prev-round-idx 2 \
  --threshold 0.3 \
  --branching-factor 10000 \
  --merge-criterion diameter \
  --reclustering-iterations 0 \
  --reclustering-extra-threshold 0.025 \
  --save-centroids \
  --save-tree \
  --slurm-mem 96G \
  --slurm-cpus 6 \
  --slurm-time 24:00:00 \
  --verbose
```

**Automatically Discovers**:
- All previous round files: `round-{prev_round_idx}-bufs*.npy`, `round-{prev_round_idx}-idxs*.pkl`
- Merges them into one final tree

**Output**: `./clustering/submit_final_round_job.sh`

---

## Complete Workflow Example

### Setup

```bash
# Assume you have 100 SMILES files, each with ~1M molecules
ls data/*.smi | wc -l          # 100 files
wc -l data/*.smi | tail -1    # ~100M total lines
```

### Step 1: Generate Initial Jobs

```bash
iChem initial-round \
  data/*.smi \
  --out-dir ./clustering \
  --files-per-job 10 \
  --threshold 0.3 \
  --branching-factor 10000 \
  --reclustering-iterations 3 \
  --verbose
```

Output:
```
[Initial Round Setup]
  Input files: 100
  Files per job: 10
  Output directory: ./clustering
  Created 10 jobs
    Job 0: 10 files, molecules 0-9999999
    Job 1: 10 files, molecules 10000000-19999999
    ...
    Job 9: 10 files, molecules 90000000-99999999

✓ Generated submission script: ./clustering/submit_initial_jobs.sh
  Run with: ./submit_initial_jobs.sh
```

### Step 2: Submit Initial Jobs

```bash
cd clustering
./submit_initial_jobs.sh
```

Output:
```
Submitted batch job 12345
Submitted batch job 12346
Submitted batch job 12347
Submitted batch job 12348
Submitted batch job 12349
Submitted batch job 12350
Submitted batch job 12351
Submitted batch job 12352
Submitted batch job 12353
Submitted batch job 12354
```

### Step 3: Monitor

```bash
watch -n 5 squeue -u $(whoami)
# or
tail -f logs/initial_*.out
```

### Step 4: Generate Midsection Jobs

After all initial jobs complete:

```bash
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 2 \
  --bin-size 5 \
  --reclustering-iterations 0 \
  --verbose
```

Output:
```
[Midsection Round 2 Setup]
  Output directory: ./clustering
  Bin size: 5
  Reading from: round-1-* files
  Found 10 buffer/index file pairs
  Created 2 batches
    Batch 0: 5 pairs
    Batch 1: 5 pairs

✓ Generated submission script: ./clustering/submit_midsection_round_2_jobs.sh
  Run with: ./submit_midsection_round_2_jobs.sh
```

### Step 5: Submit Midsection Jobs

```bash
./submit_midsection_round_2_jobs.sh
```

### Step 6: Generate Final Job

```bash
iChem final-round \
  --output-dir ./clustering \
  --prev-round-idx 2 \
  --save-centroids \
  --verbose
```

Output:
```
[Final Round Setup]
  Output directory: ./clustering
  Reading from: round-2-* files
  Found 2 buffer/index file pairs to merge

✓ Generated submission script: ./clustering/submit_final_round_job.sh
  Run with: ./submit_final_round_job.sh
```

### Step 7: Submit Final Job

```bash
./submit_final_round_job.sh
```

### Step 8: Results

```bash
ls -lh clustering/
# clusters.pkl                      # Final output
# cluster-centroids-packed.pkl      # Centroids
# bitbirch.pkl                      # Tree (if requested)
# round-*.npy/pkl                   # Intermediate (optional cleanup)
# logs/                             # All job logs
# submit_initial_jobs.sh            # Generated scripts
# submit_midsection_round_2_jobs.sh
# submit_final_round_job.sh
```

---

## Output Directory Structure

After all phases complete:

```
clustering/
├── round-1-bufs.label-*.npy           # Initial round buffers
├── round-1-idxs.label-*.pkl           # Initial round indices
├── round-2-bufs.label-*.npy           # Midsection round buffers (if kept)
├── round-2-idxs.label-*.pkl           # Midsection round indices (if kept)
├── clusters.pkl                       # FINAL: Molecule ID → Cluster ID
├── cluster-centroids-packed.pkl       # FINAL: Cluster centroids
├── bitbirch.pkl                       # FINAL: Tree object (optional)
├── logs/
│   ├── initial_00.out/err
│   ├── initial_01.out/err
│   ├── ...
│   ├── initial_09.out/err
│   ├── midsection_2_00.out/err
│   ├── midsection_2_01.out/err
│   ├── final_round.out/err
├── submit_initial_jobs.sh             # Generated submission scripts
├── submit_midsection_round_2_jobs.sh
└── submit_final_round_job.sh
```

---

## Key Parameters Explained

### `--files-per-job` (Initial Round Only)

**Definition**: Number of .smi/.smi.gz files to process per initial job.

**Example**:
- 100 .smi files with 1M molecules each
- `--files-per-job 10` → 10 jobs
- Each job processes 10 files = 10M molecules

**Recommendation**:
- Aim for **10-100M molecules per job**
- If files are 1M each → use 10-100
- If files are 10M each → use 1-10
- If files are 100M each → use 1

### `--bin-size` (Midsection Round Only)

**Definition**: Number of buffer/index file pairs per midsection batch.

**Example**:
- 10 initial jobs → 10 buffer/index pairs
- `--bin-size 5` → 2 midsection jobs
- Job 0 processes pairs 0-4
- Job 1 processes pairs 5-9

**Recommendation**:
- Default 5 is good for most cases
- Larger (e.g., 10) = fewer jobs but longer per job
- Smaller (e.g., 2) = more jobs but faster per job

### `--round-idx` (Midsection Round)

**Definition**: Current round index (reads from `round-{round_idx-1}-*` files).

**Example**:
- After initial round completes, files are `round-1-*`
- First midsection: `--round-idx 2` (reads round-1-*, writes round-2-*)
- Second midsection: `--round-idx 3` (reads round-2-*, writes round-3-*)

### `--prev-round-idx` (Final Round)

**Definition**: Previous round index (reads from `round-{prev_round_idx}-*` files).

**Example**:
- If last midsection wrote `round-2-*`, use `--prev-round-idx 2`

---

## Troubleshooting

### Initial Round Jobs Fail

**Check logs**:
```bash
cat clustering/logs/initial_00.err
cat clustering/logs/initial_00.out
```

**Common issues**:
- **SMILES parsing errors**: Some molecules invalid
  - Solution: Review error message, reduce `--n-bits` or adjust fingerprinting
- **Memory exceeded**: 16GB not enough for batch size
  - Solution: Decrease `--files-per-job`
- **File not found**: Paths in script are wrong
  - Solution: Regenerate with absolute paths

### Midsection Jobs Fail

**Check logs**:
```bash
cat clustering/logs/midsection_2_00.err
```

**Common issues**:
- **No files found**: Previous round didn't complete
  - Solution: Verify `round-1-*.npy` and `round-1-*.pkl` exist
- **Mismatched buffer/index counts**: Some jobs failed
  - Solution: Check if all initial jobs completed successfully

### Final Job Fails

**Check logs**:
```bash
cat clustering/logs/final_round.err
```

**Common issues**:
- **Insufficient memory**: Tree consolidation exceeded 96GB
  - Solution: Increase `--slurm-mem`
- **Timeout**: Job took >24 hours
  - Solution: Increase `--slurm-time`

### Restart After Failure

Intermediate files are preserved. To restart from a failed round:

```bash
# Fix the issue (e.g., adjust parameters)
# Regenerate the submission script
iChem midsection-round --output-dir ./clustering --round-idx 2 [fixed options]

# Resubmit
./clustering/submit_midsection_round_2_jobs.sh
```

---

## Performance Tips

### 1. Optimal Batch Sizing

Aim for ~20-50M molecules per initial job:
```bash
# If each file = 1M molecules
--files-per-job 20-50

# If each file = 10M molecules
--files-per-job 2-5

# If each file = 50M molecules
--files-per-job 1
```

### 2. Parallel Execution

More initial jobs = faster initial round (but more SLURM jobs):
- 100 small files + `--files-per-job 1` = 100 parallel jobs (fast)
- 100 small files + `--files-per-job 10` = 10 parallel jobs (slower, but simpler)

### 3. Threshold Tuning

Higher threshold = fewer clusters (faster):
- `--threshold 0.3` → many clusters (slower, more detailed)
- `--threshold 0.5` → fewer clusters (faster, coarser)

Test on small sample first to find good threshold.

### 4. Disable Unnecessary Reclustering

Midsection/final rounds rarely need reclustering:
```bash
# Initial: do reclustering (default 3 iterations)
iChem initial-round ... --reclustering-iterations 3

# Midsection: skip (default 0)
iChem midsection-round ... --reclustering-iterations 0

# Final: skip (default 0)
iChem final-round ... --reclustering-iterations 0
```

### 5. Use Local Storage

Store intermediate files on fast local disk, not network storage:
```bash
iChem initial-round /network/data/*.smi --out-dir /local/scratch/clustering
```

---

## Memory Efficiency

System is designed to minimize memory usage:

1. **Streaming FP save**: NumPy arrays written to disk without full materialization
2. **Progressive merging**: Buffers consolidated in phases, not all at once
3. **Intermediate cleanup**: Round files auto-deleted (optional)
4. **Tree pruning**: Internal nodes removed before serialization

**Typical memory profile** for 100M molecules:
- Initial: 16GB × ~10 jobs = distributed across HPC
- Midsection: 48GB × ~2 jobs
- Final: 96GB × 1 job
- **Peak**: 96GB (single final job, not cumulative)

---

## Implementation Details

### File Modules

**Generator modules** (run on login node):
- `_hpc_initial_submit.py` → generates `submit_initial_jobs.sh`
- `_hpc_midsection_submit.py` → generates `submit_midsection_round_*.sh`
- `_hpc_final_submit.py` → generates `submit_final_round_job.sh`

**Worker modules** (run in SLURM jobs):
- `_hpc_initial.py` → initial clustering worker
- `_hpc_midsection.py` → midsection merge worker
- `_hpc_final.py` → final consolidation worker

**Configuration**:
- `_config.py` → centralized defaults (threshold, branching_factor, SLURM resources, etc.)

### CLI Integration

**New commands** in `cli.py`:
- `iChem initial-round` → calls `_run_initial_round()`
- `iChem midsection-round` → calls `_run_midsection_round()`
- `iChem final-round` → calls `_run_final_round()`

Each command:
1. Calls corresponding generator module
2. Generates submission script
3. Prints instructions

---

## Advantages Over Traditional Multiround

| Aspect | Traditional | HPC Version |
|--------|-------------|-------------|
| Input | Pre-computed .npy fingerprints | Raw .smi/.smi.gz files |
| FP Generation | Bottleneck on single machine | Parallelized across jobs |
| Submission | Python multiprocessing | SLURM jobs with dependencies |
| Scalability | Limited by node memory | Distributed across cluster |
| Job Control | Automatic, opaque | Explicit, transparent |
| Monitoring | Python progress bars | SLURM `squeue`, job logs |
| Restartability | No (restart from scratch) | Yes (restart from failed phase) |

---

## See Also

- BitBirch documentation: `bblean` package
- Standard multiround: `multiround_reclustering.py`
- Configuration defaults: `_config.py`
