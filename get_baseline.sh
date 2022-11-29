#!/bin/bash

#SBATCH --output=baseline-%j.out
#SBATCH --job-name="baselines"
#SBATCH --time=00:30:00     # walltime
#SBATCH --gres=gpu
#SBATCH -p GPU

source /usr/local/anaconda3/etc/profile.d/conda.sh


conda activate sklearn-env

srun python model_run.py --baseline True --dataset libri-wav2vec-random | grep \]\] > ridge-results/baselines-libri-random.out

srun python model_run.py --baseline True --dataset scc-wav2vec-random | grep \]\] > ridge-results/baselines-scc-random.out
