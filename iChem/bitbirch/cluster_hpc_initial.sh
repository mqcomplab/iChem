#!/bin/bash
# Submit one SLURM job per ZINC_<number>.smi.gz file to run fps_clustering.py
#
# Usage:
#   ./sub_batch.sh <input_dir> <start_index> <end_index> [threshold] [conda_env]
#
# Example:
#   ./sub_batch.sh /blue/rmirandaquintana/klopezperez/DELs/smi_gz 1 1000 0.3 iChem

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_dir> <start_index> <end_index> [threshold] [conda_env]"
    echo ""
    echo "Arguments:"
    echo "  input_dir    : Directory containing ZINC_<number>.smi.gz files"
    echo "  start_index  : Starting file index (inclusive)"
    echo "  end_index    : Ending file index (inclusive)"
    echo "  threshold    : Optional clustering threshold (default: 0.3)"
    echo "  conda_env    : Optional conda environment name (default: iChem)"
    exit 1
fi

INPUT_DIR="$1"
START_INDEX="$2"
END_INDEX="$3"
THRESHOLD="${4:-0.3}"
CONDA_ENV="${5:-iChem}"

if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]] || [ "$START_INDEX" -le 0 ]; then
    echo "Error: start_index must be a positive integer. Got: $START_INDEX"
    exit 1
fi

if ! [[ "$END_INDEX" =~ ^[0-9]+$ ]] || [ "$END_INDEX" -le 0 ]; then
    echo "Error: end_index must be a positive integer. Got: $END_INDEX"
    exit 1
fi

if [ "$END_INDEX" -lt "$START_INDEX" ]; then
    echo "Error: end_index ($END_INDEX) must be greater than or equal to start_index ($START_INDEX)"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: input directory does not exist: $INPUT_DIR"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$INPUT_DIR/slurm_logs"

mkdir -p "$LOG_DIR"

echo "========================================"
echo "Submitting fps_clustering jobs"
echo "========================================"
echo "Input directory: $INPUT_DIR"
echo "Start index: $START_INDEX"
echo "End index: $END_INDEX"
echo "Threshold: $THRESHOLD"
echo "Conda environment: $CONDA_ENV"
echo "Logs: $LOG_DIR"
echo ""

job_count=0
missing_count=0

for ((idx=START_INDEX; idx<=END_INDEX; idx++)); do
    smi_file="$INPUT_DIR/ZINC_${idx}.smi.gz"

    if [ ! -f "$smi_file" ]; then
        echo "Skipping missing file: $smi_file"
        missing_count=$((missing_count + 1))
        continue
    fi

    base_name="$(basename "$smi_file" .smi.gz)"
    job_name="fps_${base_name}"
    log_path="$LOG_DIR/${job_name}_%j.log"

    job_id=$(sbatch <<EOF | awk '{print $NF}'
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=${log_path}

module load conda
conda activate ${CONDA_ENV}

python ${SCRIPT_DIR}/fps_clustering.py --smi_path "${smi_file}" --threshold "${THRESHOLD}"
EOF
)

    echo "Submitted $job_name (ID: $job_id)"
    job_count=$((job_count + 1))
done

echo ""
echo "========================================"
echo "Submitted $job_count jobs"
echo "Skipped missing files: $missing_count"
echo "Monitor with: squeue -u \$USER"
echo "Logs in: $LOG_DIR"
echo "========================================"