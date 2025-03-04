#!/bin/bash

service munge start
slurmd -N $SLURM_NODENAME

tail -f /dev/null
