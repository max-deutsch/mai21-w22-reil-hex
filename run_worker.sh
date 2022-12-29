#!/bin/sh
#SBATCH --job-name=worker-hex-camp
#SBATCH --output=/home/ai21m012/hex/slurm/worker-hex-camp%A_%a.out


. /opt/conda/etc/profile.d/conda.sh
conda activate myenv

srun python worker.py