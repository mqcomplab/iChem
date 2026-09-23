# HPC multi-round clustering for ultra-large libraries

This guide documents the three SLURM-oriented commands exposed by
`iChem/cli.py`:

1. `initial-round`: split the input SMILES files into independent jobs and
   create round-1 BitBirch buffers.
2. `midsection-round`: merge groups of buffer/index pairs into the next round.
   Repeat this command when one merge layer is not enough.
3. `final-round`: merge every pair from the last round and write the final
   cluster assignments.

The commands **generate submission scripts**; they do not submit jobs
themselves. Run the generated scripts with `bash` on a login node or other
node from which `sbatch` is available.

## End-to-end workflow

Assume the input files are in `data/` and that `clustering/` is shared by the
SLURM jobs:

```bash
# Generate initial-round submission script(s)
iChem initial-round data/*.smi.gz \
  --out-dir ./clustering \
  --files-per-job 10 \
  --verbose

# Submit every generated initial script
bash ./clustering/submit_initial_jobs.sh
# If max-jobs-per-script caused multiple scripts, run each one:
# bash ./clustering/submit_initial_jobs_1.sh
# bash ./clustering/submit_initial_jobs_2.sh

# Wait until every initial job has completed successfully.
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 2 \
  --bin-size 5 \
  --verbose
bash ./clustering/submit_midsection_round_2_jobs.sh

# Optional additional merge layer. Wait for round 2 to finish first.
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 3 \
  --bin-size 5
bash ./clustering/submit_midsection_round_3_jobs.sh

# Wait for the last midsection jobs, then generate and submit one final job.
iChem final-round \
  --output-dir ./clustering \
  --prev-round-idx 3 \
  --save-centroids \
  --save-npy \
  --verbose
bash ./clustering/submit_final_round_job.sh
```

If round 2 is the last midsection, use `--prev-round-idx 2` instead. Do not
generate a later round until all jobs that produce the preceding round have
finished.

### Running generated `.sh` files

The generators create executable scripts, but `bash` works regardless of the
file mode:

```bash
bash /absolute/path/to/clustering/submit_initial_jobs.sh
bash /absolute/path/to/clustering/submit_midsection_round_2_jobs.sh
bash /absolute/path/to/clustering/submit_final_round_job.sh
```

Each generated script creates a temporary SLURM job script, runs `sbatch`, and
removes the temporary script. The submitted job loads the conda module,
activates the `iChem` environment, and writes a `%j`-suffixed log under
`<out-dir>/logs/`. Check the queue and logs before starting the next phase:

```bash
squeue -u "$USER"
ls -lh clustering/logs/
tail -f clustering/logs/initial_00_12345.log
```

When `--max-jobs-per-script` is exceeded, the initial and midsection
generators return multiple scripts. Run **all** of them; the final generator
always creates exactly one script.

## Round outputs and data flow

### Initial round

`initial-round` reads one or more `.smi`/`.smi.gz` files, assigns global
molecule indices in sorted input-file order, generates fingerprints, runs
BitBirch independently for each file batch, and writes paired files:

```text
round-1-bufs*.npy   # BitBirch fingerprint buffers
round-1-idxs*.pkl   # molecule indices corresponding to each buffer
```

Both files in every pair are required by the next round. The output directory
also contains `submit_initial_jobs*.sh`, `.job_*` files while a submission
script is running, and `logs/`.

With `--result-base-dir`, the generated worker jobs write their round results
to that directory instead of `--out-dir`; use the result directory as
`--output-dir` for subsequent commands. Submission scripts and logs remain
under `--out-dir`.

### Midsection round

For `--round-idx N`, the generator reads
`round-(N-1)-bufs*.npy` and `round-(N-1)-idxs*.pkl`, groups pairs into batches,
and creates one SLURM job per batch. Each job writes:

```text
round-N-bufs*.npy
round-N-idxs*.pkl
```

The previous-round pair files processed by a successful midsection job are
deleted. This reduces disk usage, but means the source files must be restored
from backup or regenerated if a restart is needed. `--bin-size` controls how
many pairs each job loads; lower it if jobs run out of memory.

### Final round

`final-round --prev-round-idx N` reads all pairs from `round-N-*` and submits
one consolidation job. Depending on the save flags, it writes:

```text
clusters.pkl                       # cluster -> global molecule IDs
cluster-centroids-packed.pkl       # packed centroids (with --save-centroids)
bitbirch.pkl                       # full tree (with --save-tree)
clusters/cluster_<id>.npy          # one cluster file (with --save-npy)
```

When `--save-npy` is used, cluster IDs are written as separate NumPy files.
`clusters.pkl` is written as well. The final worker deletes all
`round-*-bufs*.npy` and `round-*-idxs*.pkl` files after saving results.

## `iChem initial-round`

```text
iChem initial-round INPUT [INPUT ...] --out-dir OUT_DIR [options]
```

### Inputs and job-generation options

| Option | Meaning and default |
|---|---|
| `INPUT` | One or more `.smi` or `.smi.gz` files. Files are sorted before global indices are assigned. |
| `--out-dir PATH` | **Required.** Directory for generated scripts and logs; created if needed. |
| `--files-per-job N` | Number of SMILES files handled by each initial job. Default: `FILES_PER_JOB` (`10`). |
| `--max-jobs-per-script N` | Maximum submitted jobs per generated shell script. Default: `MAX_JOBS_PER_SCRIPT` (`2000`). |
| `--result-base-dir PATH` | Optional directory for worker round outputs. Without it, workers use `--out-dir`. |
| `--code-ids` | Treat records containing SMILES and ZINC IDs (comma, tab, or space separated) as packed molecule indices. Off by default. |
| `--verbose` / `--no-verbose` | Print setup details and generated script paths. Default: off. |

### BitBirch and fingerprint options

| Option | Meaning and default |
|---|---|
| `--threshold VALUE` | BitBirch similarity/distance threshold. Default: `0.3`. |
| `--branching-factor N` | BitBirch branching factor. Default: `10000`. |
| `--merge-criterion VALUE` | BitBirch merge criterion. Default: `diameter` (the implementation also supports its configured alternatives). |
| `--fp-type VALUE` | Fingerprint type. Default: `ECFP4`; use a type supported by the fingerprint utilities. |
| `--n-bits N` | Fingerprint length. Default: `2048`. |
| `--reclustering-iterations N` | Initial-job reclustering passes. Default: `3`. |
| `--reclustering-extra-threshold VALUE` | Extra threshold used during reclustering. Default: `0.025`. |

### SLURM options

| Option | Meaning and default |
|---|---|
| `--slurm-mem VALUE` | Memory requested per job. Default: `16G`. |
| `--slurm-cpus N` | CPUs requested per job. Default: `1`. |
| `--slurm-time VALUE` | SLURM time limit per job. Default: `24:00:00`. |
| `--slurm-partition VALUE` | Optional SLURM partition. Default: empty, which uses the cluster default. |

For initial-round sizing, reduce `--files-per-job` when a job exceeds its
memory limit; increase it only when the requested memory can hold the larger
batch. Input paths are embedded in the generated scripts, so use paths visible
from the compute nodes (absolute paths are safest).

## `iChem midsection-round`

```text
iChem midsection-round --output-dir OUT_DIR --round-idx N [options]
```

| Option | Meaning and default |
|---|---|
| `--output-dir PATH` | **Required.** Directory containing the previous round's buffer/index pairs and receiving the next round. |
| `--round-idx N` | **Required.** Reads round `N-1` and writes round `N`. The first midsection is `2`. |
| `--bin-size N` | Buffer/index pairs per job. Default: `5`. Lower values reduce per-job memory. |
| `--max-jobs-per-script N` | Maximum jobs per generated script. Default: `2000`. |
| `--threshold VALUE` | BitBirch threshold for this round. Default: configured threshold (`0.3`) when omitted. |
| `--branching-factor N` | BitBirch branching factor for this round. Default: configured value (`10000`) when omitted. |
| `--merge-criterion VALUE` | BitBirch merge criterion. Default: `diameter`. |
| `--reclustering-iterations N` | Reclustering passes. Default: `0`. |
| `--reclustering-extra-threshold VALUE` | Extra reclustering threshold. Default: `0.025`. |
| `--slurm-mem VALUE` | Memory per midsection job. Default: `48G`. |
| `--slurm-cpus N` | CPUs per midsection job. Default: `3`. |
| `--slurm-time VALUE` | Time limit per job. Default: `24:00:00`. |
| `--slurm-partition VALUE` | Optional SLURM partition; empty uses the cluster default. |
| `--verbose` / `--no-verbose` | Print discovered pairs, batches, and script paths. Default: off. |

The generator fails if the previous round has no pairs or if the number of
buffer files and index files differs. It sorts pairs by fingerprint width
within each batch before embedding them in the job script.

## `iChem final-round`

```text
iChem final-round --output-dir OUT_DIR --prev-round-idx N [options]
```

| Option | Meaning and default |
|---|---|
| `--output-dir PATH` | **Required.** Directory containing the final intermediate round and receiving final outputs. |
| `--prev-round-idx N` | **Required.** Reads all `round-N-bufs*.npy`/`round-N-idxs*.pkl` pairs. |
| `--threshold VALUE` | BitBirch threshold for consolidation. Default: configured threshold (`0.3`) when omitted. |
| `--branching-factor N` | BitBirch branching factor. Default: configured value (`10000`) when omitted. |
| `--merge-criterion VALUE` | BitBirch merge criterion. Default: `diameter`. |
| `--reclustering-iterations N` | Final reclustering passes. Default: `0`. |
| `--reclustering-extra-threshold VALUE` | Extra reclustering threshold. Default: `0.025`. |
| `--save-centroids` / `--no-save-centroids` | Save centroids and cluster assignments. Default: on (`True`). |
| `--save-tree` / `--no-save-tree` | Save `bitbirch.pkl`. Default: off. |
| `--save-npy` / `--no-save-npy` | Save one `clusters/cluster_<id>.npy` per cluster. Default: off. |
| `--slurm-mem VALUE` | Memory for the single final job. Default: `96G`. |
| `--slurm-cpus N` | CPUs for the final job. Default: `6`. |
| `--slurm-time VALUE` | Time limit. Default: `24:00:00`. |
| `--slurm-partition VALUE` | Optional SLURM partition; empty uses the cluster default. |
| `--verbose` / `--no-verbose` | Print discovered pairs and the generated script path. Default: off. |

The final job is intentionally a single SLURM job because it consolidates all
remaining buffers into one tree. Increase `--slurm-mem` or `--slurm-time` for
larger libraries rather than creating multiple final scripts.

## Monitoring and restart precautions

Before moving to the next phase, verify that every submitted job completed
successfully and that the expected pair counts are present:

```bash
ls clustering/round-1-bufs*.npy | wc -l
ls clustering/round-1-idxs*.pkl | wc -l
grep -R "Complete" clustering/logs/
```

Midsection jobs delete only the input pairs they successfully process; the
final job deletes every round buffer/index pair after saving results. Keep a
backup of intermediate files if the library cannot be regenerated:

```bash
cp -a clustering clustering.before-final
```

If a phase fails, inspect its log, correct the resource or input problem,
restore any deleted intermediate pairs, regenerate that phase's submission
script, and submit the script again. Never launch a later round while an
earlier round is still writing its output.
