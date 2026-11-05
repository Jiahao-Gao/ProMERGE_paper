#!/bin/bash
#SBATCH --job-name=coemb_promergesubtract
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
    --cell_base  HEK293T \
    --cell_query HCT116 \
    --disentangle_method subtract \
    --save_update_epochs
