#!/bin/sh
#PBS -l nodes=1
#PBS -l walltime=100:00:00
#PBS -j oe
#PBS -m abe
#PBS -N atax

cd $PBS_O_WORKDIR

orcc -v atax2.src2.c > atax2.src2.rs.data

#matlab -nodesktop -nosplash -nojvm -r "randomsearchd;quit;" 

