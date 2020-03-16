#!/bin/bash -l
##SBATCH --job-name="tinyegl"
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --constraint=gpu
##SBATCH --time=00:04:00


unset DISPLAY

export LD_LIBRARY_PATH=/opt/cray/nvidia/default/lib64/:$LD_LIBRARY_PATH

#srun strace -o $SCRATCH/strace.txt /users/jfavre/Projects/EGL/code-samples/posts/egl_OpenGl_without_Xserver/tinyegl

#srun ./tinyegl
./tinyegl
