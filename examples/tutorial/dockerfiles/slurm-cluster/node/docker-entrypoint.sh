#!/bin/bash

export SLURM_CPUS_ON_NODE=$(cat /proc/cpuinfo | grep processor | wc -l)
sudo sed -i "s/REPLACE_IT/CPUs=${SLURM_CPUS_ON_NODE}/g" /etc/slurm-llnl/slurm.conf

sudo cp /scratch/munge.key /etc/munge/munge.key
sudo service munge start
sudo slurmd -N $SLURM_NODENAME

tail -f /dev/null
