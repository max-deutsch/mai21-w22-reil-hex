#!/bin/sh
#SBATCH --job-name=train-hex-camp
#SBATCH --output=/home/ai21m012/hex/slurm/train-hex-camp.out

. /opt/conda/etc/profile.d/conda.sh
conda activate myenv

python train.py