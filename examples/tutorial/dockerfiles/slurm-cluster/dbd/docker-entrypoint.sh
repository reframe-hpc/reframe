#!/bin/bash

cp /slurm.conf /etc/slurm/slurm.conf
cp /slurmdbd.conf /etc/slurm/slurmdbd.conf
cp /cgroup.conf /etc/slurm/cgroup.conf

chown slurm:slurm /etc/slurm/slurmdbd.conf /etc/slurm/slurm.conf /etc/slurm/cgroup.conf
chmod +x /etc/slurm
chmod +r /etc/slurm/slurm.conf
chmod 600 /etc/slurm/slurmdbd.conf

NUM_CPUS=$(cat /proc/cpuinfo | grep processor | wc -l)
export SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:=$NUM_CPUS}
sed -i "s/REPLACE_IT/CPUs=${SLURM_CPUS_ON_NODE}/g" /etc/slurm/slurm.conf

service munge start
service slurmdbd start

tail -f /dev/null
