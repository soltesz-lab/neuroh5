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

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

aprun -n 512 ./neurotrees_scatter_read -a -n Synapse_Attributes -i 128  \
      /projects/sciteam/baef/DGC_forest_syns_20170111.h5


