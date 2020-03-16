#!/bin/bash
#SBATCH --job-name="tinyegl"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
##SBATCH --reservation=maint
#SBATCH --time=00:04:00
##SBATCH --mail-type=ALL
##SBATCH --mail-user=jfavre@cscs.ch
#SBATCH --output=/users/jfavre/tinyegl.ppm
#SBATCH --error=/users/jfavre/tinyegl-ERR.log

##SBATCH --constraint=startx

# the module is inherited from the shell. 
# You must load the paraview module before submitting the SLURM job
# tested Fri Apr 18 15:37:52 CEST 2014

#export DISPLAY=:0

export LD_LIBRARY_PATH=/opt/cray/nvidia/default/lib64:$LD_LIBRARY_PATH

srun -n $SLURM_NTASKS .//tinyegl
