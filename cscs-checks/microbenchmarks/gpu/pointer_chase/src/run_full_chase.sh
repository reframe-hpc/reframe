#!/bin/bash

mkdir -p data

for stride in 1 2 4 8 16 32
do
  echo "Running chase with stride of ${stride}"
  n=8
  while [ $n -le 268435456 ] # Up to 2GB worth of list (8 Bytes per node)
  do
    srun -n 1 ./pChase.x --stride $stride --num-jumps 400000 --nodes $n  > data/out_${n}_${stride}.dat 
    n=$(( $n * 2 ))
  done
done
