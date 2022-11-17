#!/bin/bash

#SBATCH --output=extract_wav2vec.out
#SBATCH --job-name="extract"
#SBATCH --time=1:30:00     # walltime
#SBATCH --gres=gpu
#SBATCH -p GPU

source /usr/local/anaconda3/etc/profile.d/conda.sh


conda activate sklearn-env

# srun python -u extract_spokencoco_embeddings.py
srun python -u extract_embeddings.py --model wav2vec --corpus librispeech