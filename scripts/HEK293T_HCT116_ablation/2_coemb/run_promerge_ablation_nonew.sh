#!/bin/bash
#SBATCH --job-name=coemb_promerge_ablation_noaux
#SBATCH --partition=nrnb-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -euo pipefail
source /home/jig049/miniforge3/etc/profile.d/conda.sh
conda activate cellmap

python ./coemb_promerge.py \
    --cell_base  HCT116 \
    --cell_query HEK293T \
    --disentangle_method CLUB \
    --lambda_replay 0 \
    --lambda_var 0 \
    --lambda_mean 0 \
    --align_inputs none \
    --run_name promerge_ablation_nonew_HCT116anchor
