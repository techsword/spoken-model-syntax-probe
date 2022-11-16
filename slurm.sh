#!/bin/bash

#SBATCH --output=slurm-%A_layer%a.out
#SBATCH --job-name="classification"
#SBATCH --time=4:00:00     # walltime
#SBATCH --array=0-11%4
#SBATCH --gres=gpu
#SBATCH -p GPU

source /usr/local/anaconda3/etc/profile.d/conda.sh

# conda config --add envs_dirs /home/gshen/.conda/envs
# conda config --add pkgs_dirs /home/gshen/.conda/pkgs

conda activate sklearn-env

# python -u classifier.py 
# counter=0
# while [ $counter -le 11 ]
# do
    # echo $counter
srun python -u model_run.py --layer $SLURM_ARRAY_TASK_ID  --model True #--num_data 10000
    # ((counter++))
# done

# echo All done

# Run 'srun python model_run.py --baseline True' to get baseline