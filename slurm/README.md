# SLURM / HPC Guide

This folder contains job scripts and a batch-submission helper for running training on an HPC cluster.

## Files

| File | Purpose |
|---|---|
| `train_resnet50.slurm` | ResNet-50 training job |
| `train_vit_b16.slurm` | ViT-B/16 training job |
| `submit_all.sh` | Submit both jobs at once (with optional dependency chaining) |

---

## Storage layout

- **`/scratch/`** — large files (data, checkpoints, outputs); high-speed but may be purged periodically
- **`/home/`** — code repo and final results; persistent but limited quota

Default per-user paths used by the scripts:

| Resource | Path |
|---|---|
| Data root | `/scratch/$USER/DS5500_Data_Capstone/data/sampled_data_5k` |
| Run artifacts | `/scratch/$USER/DS5500_Data_Capstone/aigi_runs/<RUN_ID>/` |
| SLURM logs | `/scratch/$USER/DS5500_Data_Capstone/aigi_logs/` |

---

## Submitting jobs

### `sbatch` — background (recommended)

```bash
# From repo root
cd /home/$USER/DS5500_Data_Capstone/DS5500-Detecting_AI_Generated_Images
sbatch slurm/train_resnet50.slurm
sbatch slurm/train_vit_b16.slurm

# Or submit both at once
bash slurm/submit_all.sh
```

`submit_all.sh` prints the assigned job IDs and includes a commented-out template for a third job that waits for both to succeed (`--dependency=afterok`).

### `srun` — interactive session

**Stage 1 (linear probe):**
```bash
srun --partition=gpu --gres=gpu:v100-sxm2:1 --pty bash
cd /home/$USER/DS5500_Data_Capstone/DS5500-Detecting_AI_Generated_Images
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

SCRATCH_BASE=/scratch/$USER/DS5500_Data_Capstone
RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_DIR=$SCRATCH_BASE/aigi_runs/$RUN_ID
mkdir -p "$BASE_DIR/checkpoints" "$BASE_DIR/outputs" "$SCRATCH_BASE/aigi_logs"

python -u -m training.train \
    --config configs/vit_b16.yaml \
    --data_root $SCRATCH_BASE/data/sampled_data_5k \
    --num_workers 1 \
    --save_dir "$BASE_DIR/checkpoints" \
    --outputs_dir "$BASE_DIR/outputs" \
    2>&1 | tee "$SCRATCH_BASE/aigi_logs/srun-vit-stage1-$RUN_ID.log"
```

After Stage 1 finishes the script writes a `latest_vit` symlink pointing to that run's directory.

**Stage 2 (fine-tune, warm-start from Stage 1):**
```bash
srun --partition=gpu --gres=gpu:v100-sxm2:1 --pty bash
cd /home/$USER/DS5500_Data_Capstone/DS5500-Detecting_AI_Generated_Images
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

SCRATCH_BASE=/scratch/$USER/DS5500_Data_Capstone
RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_DIR=$SCRATCH_BASE/aigi_runs/$RUN_ID
mkdir -p "$BASE_DIR/checkpoints" "$BASE_DIR/outputs" "$SCRATCH_BASE/aigi_logs"

# Pick up the best checkpoint from the most recent Stage 1 run
STAGE1_CKPT=$(ls -t $SCRATCH_BASE/aigi_runs/latest_vit/checkpoints/best_model_*.pth | head -1)
echo "Warm-starting from: $STAGE1_CKPT"

python -u -m training.train \
    --config configs/vit_b16.yaml \
    --checkpoint "$STAGE1_CKPT" \
    --unfreeze_last_n_blocks 2 \
    --backbone_lr 1e-5 \
    --lr 1e-4 \
    --run_name vit-b16-ft \
    --data_root $SCRATCH_BASE/data/sampled_data_5k \
    --num_workers 1 \
    --save_dir "$BASE_DIR/checkpoints" \
    --outputs_dir "$BASE_DIR/outputs" \
    2>&1 | tee "$SCRATCH_BASE/aigi_logs/srun-vit-stage2-$RUN_ID.log"
```

To use a specific checkpoint instead of `latest_vit`, replace the `ls` line with:
```bash
STAGE1_CKPT=$SCRATCH_BASE/aigi_runs/<RUN_ID>/checkpoints/best_model_<timestamp>.pth
```

---

## Overriding hyperparameters at submit time

All scripts pass `EXTRA_ARGS` verbatim to `training/train.py`, overriding whatever the YAML config sets — no file edits required.

```bash
# Override learning rate and number of epochs
sbatch --export=ALL,EXTRA_ARGS="--epochs 20 --lr 3e-4" slurm/train_vit_b16.slurm

# Fine-tune: unfreeze last 2 backbone blocks
sbatch --export=ALL,EXTRA_ARGS="--unfreeze_last_n_blocks 2 --backbone_lr 5e-6 --run_name vit-ft" \
    slurm/train_vit_b16.slurm
```

### Grid search via `submit_all.sh`

`submit_all.sh` has a built-in grid search mode that submits one job per combination of learning rate and batch size:

```bash
# Run grid search for ViT (default)
GRID_SEARCH=1 bash slurm/submit_all.sh

# Restrict to one model
GRID_SEARCH=1 GRID_MODEL=vit      bash slurm/submit_all.sh
GRID_SEARCH=1 GRID_MODEL=resnet50 bash slurm/submit_all.sh
GRID_SEARCH=1 GRID_MODEL=both     bash slurm/submit_all.sh
```

Edit the search space at the top of the grid section in `submit_all.sh`:

```bash
LR_VALUES=(1e-3 3e-4 1e-4)
BATCH_VALUES=(32 64)
EPOCHS=20
```

Each job gets a unique `--run_name` (e.g. `vit-lr3e-4_bs32`) so outputs are easy to tell apart.

### Manual sweep (alternative)

```bash
# Hyperparameter sweep without submit_all.sh
for LR in 1e-3 3e-4 1e-4; do
    sbatch --export=ALL,EXTRA_ARGS="--lr $LR --run_name vit-lr$LR" slurm/train_vit_b16.slurm
done
```

### All overridable CLI flags

| Flag | Type | Description |
|---|---|---|
| `--config` | str | Path to YAML config file |
| `--epochs` | int | Total training epochs |
| `--batch_size` | int | Batch size for all data loaders |
| `--lr` | float | Head learning rate |
| `--backbone_lr` | float | Learning rate for unfrozen backbone layers |
| `--weight_decay` | float | AdamW weight decay |
| `--patience` | int | Early-stopping patience |
| `--run_name` | str | Name prefix for output files |
| `--unfreeze_last_n_blocks` | int | Backbone blocks to unfreeze (0 = linear probe) |
| `--num_workers` | int | DataLoader workers (1 on HPC, 0 for CPU-only) |
| `--data_root` | str | Root folder containing train/val/test sub-dirs |
| `--save_dir` | str | Checkpoints directory |
| `--outputs_dir` | str | Metrics CSVs and figures directory |

---

## Artifacts

Each run writes to `/scratch/$USER/DS5500_Data_Capstone/aigi_runs/<RUN_ID>/`:

**`checkpoints/`**
- `best_model_<timestamp>.pth`
- `test_metrics_<timestamp>.json`
- `test_preds_<timestamp>.npz`
- `config.yaml`

**`outputs/`**
- `metrics/*_history.csv` — per-epoch train/val metrics
- `figures/*.png` — training curves, confusion matrix, ROC curve

---

## Monitoring

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
tail -f /scratch/$USER/DS5500_Data_Capstone/aigi_logs/<job-name>-<jobid>.out
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'training'`**
- Cause: `python -m training.train` was run outside the repo root.
- Fix: `cd` to the repo root first, or export `PYTHONPATH`:
  ```bash
  export PYTHONPATH="/path/to/DS5500-Detecting_AI_Generated_Images${PYTHONPATH:+:$PYTHONPATH}"
  ```

**`conda: command not found` in a batch job**
- The scripts run `module purge` then `module load $ANACONDA_MODULE`. Pass the correct module name:
  ```bash
  sbatch --export=ALL,ANACONDA_MODULE=anaconda3/2024.06 slurm/train_resnet50.slurm
  ```

**Submitting from outside the repo root**
- Pass `PROJECT_ROOT` explicitly:
  ```bash
  sbatch --export=ALL,PROJECT_ROOT=/home/$USER/DS5500_Data_Capstone/DS5500-Detecting_AI_Generated_Images \
      /path/to/slurm/train_resnet50.slurm
  ```
