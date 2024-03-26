#!/bin/bash

trap exit 0 INT

while [ ! -f /scratch/munge.key ]
do
  sleep 1
done

sudo cp /scratch/munge.key /etc/munge/munge.key
sudo service munge start
sudo sed -i "s/REPLACE_IT/CPUs=${SLURM_CPUS_ON_NODE}/g" /etc/slurm-llnl/slurm.conf

echo "Container up and running: "
echo "==> Run 'docker exec -it <container-id> /bin/bash' to run interactively"
echo "==> Press Ctrl-C to exit"
sleep infinity &
wait
