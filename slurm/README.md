# Slurm templates

This folder contains templates for running training in the background with Slurm.

## Files

- train_resnet50.slurm
- train_vit_b16.slurm

## Usage

From repo root:

```bash
sbatch slurm/train_resnet50.slurm
sbatch slurm/train_vit_b16.slurm
```

Both templates now default to:

- `ANACONDA_MODULE=anaconda3/2024.06`
- `ENV_NAME=ds5500-aigi`

Override variables at submit time:

```bash
sbatch --export=ALL,ENV_NAME=ds5500-aigi slurm/train_resnet50.slurm
sbatch --export=ALL,ENV_NAME=ds5500-aigi,CONFIG_PATH=configs/vit_b16.local.yaml slurm/train_vit_b16.slurm
```

## Interactive sessions

When running on a compute node interactively (not via `sbatch`), set `PYTHONPATH`
manually so the local packages can be found regardless of your working directory:

```bash
cd /path/to/DS5500-Detecting_AI_Generated_Images
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m training.train --config configs/resnet50.yaml
```

If needed, you can also override the module and project root:

```bash
sbatch --export=ALL,ANACONDA_MODULE=anaconda3/2024.06,PROJECT_ROOT=/home/$USER/DS5500_Data_Capstone/DS5500-Detecting_AI_Generated_Images slurm/train_resnet50.slurm
```

To store run artifacts and Slurm logs under a custom scratch base:

```bash
sbatch --export=ALL,SCRATCH_BASE=/scratch/$USER/DS5500_Data_Capstone slurm/train_resnet50.slurm
```

Check status and logs:

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
tail -f /scratch/$USER/DS5500_Data_Capstone/aigi_logs/<job-name>-<jobid>.out
```

## Artifacts

Each run gets a unique timestamped directory:

- /scratch/$USER/DS5500_Data_Capstone/aigi_runs/<RUN_ID>/checkpoints
- /scratch/$USER/DS5500_Data_Capstone/aigi_runs/<RUN_ID>/outputs

Inside outputs you will have:

- train_console.log (full console log)
- metrics/*_history.csv (training history)
- figures/*.png (training curves, confusion matrix, ROC)

Inside checkpoints you will have:

- best_model_<timestamp>.pth
- test_metrics_<timestamp>.json
- test_preds_<timestamp>.npz
- config.yaml

## Notes

- The scripts run `module purge` then `module load $ANACONDA_MODULE` to ensure `conda` exists in batch jobs.
- If `conda` is still not found on your cluster, ask admins for the correct module name, then pass it via `ANACONDA_MODULE=...`.
- If your cluster does not provide `vit_b16.local.yaml`, copy from `configs/vit_b16.local.yaml.example` and update paths.
