#!/bin/bash
#SBATCH --job-name=n2v
#SBATCH --partition=nrnb-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --array=0-599%200

set -euo pipefail
source /home/jig049/miniforge3/etc/profile.d/conda.sh
conda activate cellmap

# 150 settings x 4 networks
NETWORKS=(BioPlex-HEK293T BioPlex-HCT116 SECMX-HEK293T SECMX-HCT116)
CONFIG_ID=$(( SLURM_ARRAY_TASK_ID % 150 ))
NETWORK=${NETWORKS[$(( SLURM_ARRAY_TASK_ID / 150 % 4 ))]}

python ./n2v_gridsearch.py \
    --infile ../../../result/HEK293T_HCT116/input/${NETWORK}.ppi \
    --outdir ../../../result/HEK293T_HCT116/n2v_gridsearch \
    --config_id ${CONFIG_ID}
