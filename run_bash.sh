#!/bin/bash

counter=0
while [ $counter -le 11 ]
do
    echo $counter
srun python -u model_run.py --layer $counter --model True --dataset $1 --modelname $2 >> $1-$2.out
    ((counter++))
done

srun python model_run.py --baseline True --dataset $1 --modelname $2 >> $1-$2.out

echo All done
