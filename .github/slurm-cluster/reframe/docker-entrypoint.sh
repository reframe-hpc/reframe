#!/bin/bash

trap exit 0 INT

while [ ! -f /scratch/munge.key ]
do
  sleep 1
done

sudo cp /scratch/munge.key /etc/munge/munge.key
sudo service munge start
sudo sed -i "s/REPLACE_IT/CPUs=${SLURM_CPUS_ON_NODE}/g" /etc/slurm-llnl/slurm.conf

# Needs to be copied in the shared home directory
cp -r /usr/local/share/reframe .
cd reframe
./bootstrap.sh

tempdir=$(mktemp -d -p /scratch)
TMPDIR=$tempdir ./test_reframe.py -v --rfm-user-config=examples/tutorial/config/cluster.py
