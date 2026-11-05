#!/bin/bash
#SBATCH --job-name=coemb_ae
#SBATCH --partition=nrnb-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=6:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -euo pipefail
source /home/jig049/miniforge3/etc/profile.d/conda.sh
conda activate cellmap

python ./coemb_autoencoder.py \
    --cell HCT116
