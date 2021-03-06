#!/bin/bash

### set the number of nodes and the number of PEs per node
#PBS -l nodes=32:ppn=16:xe
### which queue to use
#PBS -q debug
### set the wallclock time
#PBS -l walltime=0:30:00
### set the job name
#PBS -N scatter_neurotrees
### set the job stdout and stderr
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
### Get darsan profile data
#PBS -lgres=darshan

module load cray-hdf5-parallel

set -x

export PYTHONPATH=/projects/sciteam/baef/site-packages:$PYTHONPATH

cd $PBS_O_WORKDIR

aprun -n 512 ./neurotrees_scatter_read -i 64  \
      /u/sciteam/raikov/scratch/DGC_forest/DGC_forest_only.h5



