#!/bin/bash

RPT=rpt_`hostname`_${SLURM_PROCID}.csv
QRY='timestamp,gpu_name,pid,gpu_util,mem_util,max_memory_usage,time'

nvidia-smi \
--query-accounted-apps=$QRY \
--format=csv |tail -2 &> $RPT
