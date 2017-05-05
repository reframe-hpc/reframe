#!/bin/bash -l
#
#========================================
# DESCRIPTION (APPLICATION SPECIFIC)
#========================================
# The script runs the CeO2 test of Peter Larsson's VASP test suite
# It can be launched by the following scripts:
# - run_regression.sh
# - 9000-scientific_applications_launcher.sh
# - only if $USER has write permissions in $SLURM_SUBMIT_DIR: "sbatch $TESTID-$TESTNAME-$machine.sbatch"
# The test will run on the $SCRATCH of the $USER that runs this script
# The output and log file will be saved ...
#
# Input files:
#  INCAR
#  KPOINTS
#  POSCAR
#  POTCAR
# 
# Output files:
#  $outdir/$TESTID-$TESTNAME-$SLURM_JOB_ID.out
#
# Log and reference file for perfomance:
#  $logdir/$TESTID-$TESTNAME.log
#  $TESTNAME.ref
#========================================
# SLURM VARIABLES (APPLICATION SPECIFIC)
#========================================
#SBATCH --job-name=9060-vasp_mc
#SBATCH --time=00:10:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#===================================
# SETUP (GENERAL)
#===================================
TESTID="${SLURM_JOB_NAME:0:4}";
TESTNAME="${SLURM_JOB_NAME:5}";
echo -e "\n Test $TESTID: $TESTNAME";
# check if outdir is defined
if [ -z $logdir ]; then
 logdir=$SLURM_SUBMIT_DIR;
fi
# check if outdir is defined
if [ -z $outdir ]; then
 outdir=$SLURM_SUBMIT_DIR;
fi
# check if scrdir is defined
if [ -z $scrdir ]; then
 cdate=$(date "+%d-%m-%Y_%H-%M-%S");
 scrdir=$SCRATCH/$cdate/$TESTID;
 mkdir -p $scrdir;
fi
# check if machine is defined
if [ -z $machine ]; then
 machine=$(echo $HOSTNAME | sed 's/[0-9]*//g');
fi
# output file
out="OUTCAR";
# logfile with history of performance
log="$logdir/$TESTID-$TESTNAME.log";
# check if logfile exists and if it is group readable and writeable
if [ ! -f "$log" ]; then
 echo "# wall-time, performance (day/ns) and % performance difference on $SLURM_JOB_NUM_NODES nodes, $SLURM_NTASKS_PER_NODE MPI tasks per node" > $log;
# make the newly created $log file group readable and writeable
 chmod g+rw $log;
elif [ ! -r "$log" ]; then
# echo an error if $log file is not group readable and exit
 echo -e "$machine $cdate $SLURM_JOB_ID \t ERROR: log file $log is not group readable!" >> /dev/stderr;
 exit 1;
elif [ ! -w "$log" ]; then
# echo an error if $log file is not group writeable and exit
 echo -e "$machine $cdate $SLURM_JOB_ID \t ERROR: log file $log is not group writeable!" >> /dev/stderr;
 exit 1;
fi
# reference performance file
reffile="$SLURM_SUBMIT_DIR/$TESTNAME.ref";
# SLURM variables
SLURM_NTASKS=$[$SLURM_NTASKS_PER_NODE*$SLURM_JOB_NUM_NODES];
echo -e "\nRunning on $machine (job ID $SLURM_JOB_ID)";
echo -e "Log file      : $log";
echo -e "Output folder : $outdir";
echo -e "Running folder: $scrdir";
echo -e "$SLURM_JOB_NUM_NODES nodes, $SLURM_NTASKS_PER_NODE tasks per node ($SLURM_TASKS_PER_NODE), $SLURM_CPUS_PER_TASK cpus per task, $SLURM_NTASKS total tasks\n";
#========================================
# RUN AND CHECK (APPLICATION SPECIFIC)
#========================================
# copy files to $scrdir and change directory
cp INCAR KPOINTS POSCAR POTCAR $scrdir;
cd $scrdir
# OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
# set stack size unlimited
ulimit -s unlimited;
# load modules and run simulation
module load slurm;
module load VASP;
EXE="vasp_std";
srun -n $SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE -c $SLURM_CPUS_PER_TASK $EXE;
#========================================
# check output file
check=$(grep "aborting loop because EDIFF is reached" "$out" | awk '{print $7}');
if [[ "$check" != "reached" ]]; then
 echo -e "$machine $timestamp $SLURM_JOB_ID \t Job output NOT OK: test FAILED!" >> "$log";
 exit 1;
fi
#========================================
# summary of results: new time (s)
new=$(grep "Total CPU time used" "$out" | awk '{print $6}');
# reference time (s), performance difference (%)
if [ -f "$reffile" ]; then
 ref=$(grep -v '#' $reffile | grep $machine | tail -1 | sed 's/Time\=//' | awk '{printf"%.3lf\n", $(NF-1)}');
 if [ -n "$ref" ]; then
  delta=$(printf '%.3lf\n' $(echo "(1-$new/$ref)*100" | bc -l));
 else
  delta=0;
 fi
else
 delta=0;
fi
# print results on log file
echo -e "$machine $cdate $SLURM_JOB_ID \t Time=$new Difference=$delta%" >> "$log";
#========================================
# COPY OUTPUT (GENERAL)
#========================================
# copy output and log files to outdir
store="$SLURM_JOB_NAME-$SLURM_JOB_ID.out";
chmod g+rw $out;
cp -p $out $outdir/$store;
exit 0;
#========================================
