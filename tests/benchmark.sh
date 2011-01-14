#!/bin/bash

benchmark=$1
input=$2
ntrials=$3
max_threads=$4

output=${benchmark}.log
cat /dev/null > ${output}

for((n=0;n<ntrials;n++))
do
    for((i=1;i<=max_threads;i++))
    do
        echo "Running trial $n on $i threads."
        OMP_NUM_THREADS=$i ${benchmark} ${input} &>> ${output}
    done
done
