#!/bin/bash

#SBATCH --output=slurm-%A_layer%a.out
#SBATCH --job-name="model_run"
#SBATCH --time=4:00:00     # walltime
#SBATCH --array=0-11%4
#SBATCH --gres=gpu
#SBATCH -p GPU

source /usr/local/anaconda3/etc/profile.d/conda.sh


conda activate sklearn-env


srun python -u model_run.py --layer $SLURM_ARRAY_TASK_ID  --model True --dataset $1 --modelname $2 #--num_data 10000



# srun python model_run.py --baseline True --dataset $1 > slurm-$SLURM_ARRAY_JOB_ID-baseline.out

cat slurm-$SLURM_ARRAY_JOB_ID* | grep \]\] >> $1-$2.out