#!/bin/bash
#SBATCH --job-name=aigi-vit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/DS5500_Data_Capstone/aigi_logs/%x-%j.out
#SBATCH --error=/scratch/%u/DS5500_Data_Capstone/aigi_logs/%x-%j.err

set -euo pipefail

# ------------------------------
# User-configurable defaults
# Override at submit time, e.g.:
# sbatch --export=ALL,ENV_NAME=myenv,CONFIG_PATH=configs/vit_b16.local.yaml slurm/train_vit_b16.slurm
# ------------------------------
ENV_NAME=${ENV_NAME:-ds5500-aigi}
ANACONDA_MODULE=${ANACONDA_MODULE:-anaconda3/2024.06}
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/DS5500_Data_Capstone/DS5500-Detecting_AI_Generated_Images}
CONFIG_PATH=${CONFIG_PATH:-configs/vit_b16.local.yaml}
SCRATCH_BASE=${SCRATCH_BASE:-/scratch/$USER/DS5500_Data_Capstone}
DATA_ROOT=${DATA_ROOT:-/scratch/$USER/DS5500_Data_Capstone/data/sampled_data_5k}
NUM_WORKERS=${NUM_WORKERS:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}   # e.g. "--epochs 5 --lr 1e-4 --batch_size 32"

# Load conda from module system for this batch shell.
module purge
module load "$ANACONDA_MODULE"

if ! command -v conda >/dev/null 2>&1; then
  echo "[Error] conda command not found after loading module: $ANACONDA_MODULE"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "[Env] Active conda env: $ENV_NAME"
which python
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('torch_cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count(), 'device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Move to repo root and ensure it is on PYTHONPATH.
# (Required so that `python -m training.train` can locate the local packages
# regardless of the working directory at job submission time.)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH=$PYTHONPATH"

# Per-run directory to avoid overwriting artifacts.
RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_DIR=$SCRATCH_BASE/aigi_runs/$RUN_ID
SAVE_DIR=$BASE_DIR/checkpoints
OUT_DIR=$BASE_DIR/outputs
mkdir -p "$SAVE_DIR" "$OUT_DIR" "$SCRATCH_BASE/aigi_logs"

echo "RUN_ID=$RUN_ID"
echo "SAVE_DIR=$SAVE_DIR"
echo "OUT_DIR=$OUT_DIR"

python -u -m training.train \
  --config "$CONFIG_PATH" \
  --data_root "$DATA_ROOT" \
  --num_workers "$NUM_WORKERS" \
  --save_dir "$SAVE_DIR" \
  --outputs_dir "$OUT_DIR" \
  $EXTRA_ARGS \
  2>&1 | tee "$OUT_DIR/train_console.log"

echo "Training finished."
