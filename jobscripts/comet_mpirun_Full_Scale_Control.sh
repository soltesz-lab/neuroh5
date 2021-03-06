#!/bin/bash
#
#SBATCH -J neurograph_scatter_attrs_test
#SBATCH -o ./results/neurograph_scatter_attrs_test.%j.o
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=16
#SBATCH -p compute
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

module load hdf5

#export input_file=/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_Control_MPP.h5
export input_file=/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/dentate_Full_Scale_GC_20170902.h5

ibrun ./build/neurograph_scatter -i 128 -a \
 -o /oasis/scratch/comet/iraikov/temp_project/dentate_Full_Scale_Control $input_file


