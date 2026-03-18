#!/bin/bash
# submit_all.sh — Submit training jobs to SLURM.
#
# NORMAL MODE (default): submit one ResNet-50 and one ViT-B/16 baseline run.
#   bash slurm/submit_all.sh
#
# GRID SEARCH MODE: submit one job per hyperparameter combination.
#   GRID_SEARCH=1 bash slurm/submit_all.sh
#
#   Optionally restrict to one model:
#   GRID_SEARCH=1 GRID_MODEL=resnet50 bash slurm/submit_all.sh
#   GRID_SEARCH=1 GRID_MODEL=vit      bash slurm/submit_all.sh

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
COMMON="--export=ALL,PROJECT_ROOT=$REPO_ROOT,ENV_NAME=ds5500-aigi"
GRID_SEARCH=${GRID_SEARCH:-0}
GRID_MODEL=${GRID_MODEL:-vit}    # resnet50 | vit | both

echo "=== Submitting AIGI training jobs ==="
echo "Repo root : $REPO_ROOT"
echo "Mode      : $([ "$GRID_SEARCH" = "1" ] && echo "GRID SEARCH (model=$GRID_MODEL)" || echo "NORMAL")"
echo ""

# ==================================================================
# NORMAL MODE
# ==================================================================
if [ "$GRID_SEARCH" != "1" ]; then

  # Job 1: ResNet-50 linear probe (commented out — focusing on ViT)
  # JOB1=$(sbatch $COMMON \
  #   --export=ALL,PROJECT_ROOT=$REPO_ROOT,ENV_NAME=ds5500-aigi,CONFIG_PATH=configs/resnet50.yaml \
  #   slurm/train_resnet50.slurm | awk '{print $NF}')
  # echo "Submitted ResNet-50 linear probe  → job $JOB1"

  # Job 2: ViT-B/16 linear probe
  JOB2=$(sbatch $COMMON \
    --export=ALL,PROJECT_ROOT=$REPO_ROOT,ENV_NAME=ds5500-aigi,CONFIG_PATH=configs/vit_b16.yaml \
    slurm/train_vit_b16.slurm | awk '{print $NF}')
    # --dependency=afterok:$JOB1 \   # ← uncomment to make sequential
  echo "Submitted ViT-B/16 linear probe   → job $JOB2"

  # ----------------------------------------------------------------
  # Stage 2: ViT fine-tune — runs after Stage 1, warm-starts from its
  # checkpoint via the stable 'latest_vit' symlink.
  # Uncomment and adjust hyperparameters when ready.
  # ----------------------------------------------------------------
  # SCRATCH_BASE=${SCRATCH_BASE:-/scratch/$USER/DS5500_Data_Capstone}
  # STAGE1_CKPT=$(ls -t $SCRATCH_BASE/aigi_runs/latest_vit/checkpoints/best_model_*.pth 2>/dev/null | head -1)
  # JOB_FT=$(CHECKPOINT="$STAGE1_CKPT" \
  #   EXTRA_ARGS="--unfreeze_last_n_blocks 2 --backbone_lr 1e-5 --lr 1e-4 --run_name vit-b16-ft" \
  #   sbatch \
  #     --job-name=aigi-vit-ft \
  #     --dependency=afterok:$JOB2 \
  #     --export=ALL,PROJECT_ROOT=$REPO_ROOT,ENV_NAME=ds5500-aigi,CONFIG_PATH=configs/vit_b16.yaml \
  #     slurm/train_vit_b16.slurm | awk '{print $NF}')
  # echo "Submitted ViT-B/16 fine-tune      → job $JOB_FT (after $JOB2, warm-start from Stage 1)"

# ==================================================================
# GRID SEARCH MODE
# Edit the arrays below, then run:  GRID_SEARCH=1 bash slurm/submit_all.sh
# ==================================================================
else

  # ---- define the search space ----
  LR_VALUES=(1e-2 1e-3 1e-4)
  BATCH_VALUES=64
  EPOCHS=20                      # fixed for all sweep jobs
  # ---------------------------------

  COUNT=0

  for LR in "${LR_VALUES[@]}"; do
    for BS in "${BATCH_VALUES[@]}"; do

      # build a short tag like "lr1e-3_bs32"
      TAG="lr${LR}_bs${BS}"

      if [ "$GRID_MODEL" = "resnet50" ] || [ "$GRID_MODEL" = "both" ]; then
        RUN_NAME="resnet50-${TAG}"
        EXTRA="--epochs $EPOCHS --lr $LR --batch_size $BS --run_name $RUN_NAME"
        JID=$(EXTRA_ARGS="$EXTRA" sbatch \
          --job-name="$RUN_NAME" \
          --export=ALL,PROJECT_ROOT=$REPO_ROOT,ENV_NAME=ds5500-aigi,CONFIG_PATH=configs/resnet50.yaml \
          slurm/train_resnet50.slurm | awk '{print $NF}')
        echo "  ResNet-50  $TAG  → job $JID"
        COUNT=$((COUNT + 1))
      fi

      if [ "$GRID_MODEL" = "vit" ] || [ "$GRID_MODEL" = "both" ]; then
        RUN_NAME="vit-${TAG}"
        EXTRA="--epochs $EPOCHS --lr $LR --batch_size $BS --run_name $RUN_NAME"
        JID=$(EXTRA_ARGS="$EXTRA" sbatch \
          --job-name="$RUN_NAME" \
          --export=ALL,PROJECT_ROOT=$REPO_ROOT,ENV_NAME=ds5500-aigi,CONFIG_PATH=configs/vit_b16.yaml \
          slurm/train_vit_b16.slurm | awk '{print $NF}')
        echo "  ViT-B/16   $TAG  → job $JID"
        COUNT=$((COUNT + 1))
      fi

    done
  done

  echo ""
  echo "Submitted $COUNT jobs  (${#LR_VALUES[@]} LR × ${#BATCH_VALUES[@]} batch sizes)"

fi

# ==================================================================
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f /scratch/\$USER/DS5500_Data_Capstone/aigi_logs/<job>.out"
