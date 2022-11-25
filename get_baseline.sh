#!/bin/bash

#SBATCH --output=baseline-%j.out
#SBATCH --job-name="baselines"
#SBATCH --time=00:30:00     # walltime
#SBATCH --gres=gpu
#SBATCH -p GPU

srun python model_run.py --baseline True --dataset libri-hubert | grep \]\] > baselines-libri.out

srun python model_run.py --baseline True --dataset scc-hubert | grep \]\] > baselines-scc.out
