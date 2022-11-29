#!/bin/bash

#SBATCH --output=extract_scc_proper.out
#SBATCH --job-name="extract"
#SBATCH --gres=gpu
#SBATCH -p GPU

source /usr/local/anaconda3/etc/profile.d/conda.sh


conda activate sklearn-env


srun python -u extract_embeddings.py --model hubert --corpus spokencoco
srun python -u extract_embeddings.py --model wav2vec --corpus spokencoco

srun python -u extract_embeddings.py --model hubert --corpus librispeech
srun python -u extract_embeddings.py --model wav2vec --corpus librispeech

srun python -u extract_embeddings.py --model random --corpus spokencoco
srun python -u extract_embeddings.py --model random --corpus librispeech

srun python -u extract_fast_vgs_embeddings.py --model fast-vgs --corpus spokencoco
srun python -u extract_fast_vgs_embeddings.py --model fast-vgs --corpus librispeech
srun python -u extract_fast_vgs_embeddings.py --model fast-vgs-plus --corpus spokencoco
srun python -u extract_fast_vgs_embeddings.py --model fast-vgs-plus --corpus librispeech