#!/bin/bash

counter=0
while [ $counter -le 11 ]
do
    echo $counter
srun python -u classifier.py --layer $counter
    ((counter++))
done

echo All done
