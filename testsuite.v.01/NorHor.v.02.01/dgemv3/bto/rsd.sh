#!/bin/sh
#PBS -l nodes=1
#PBS -l walltime=100:00:00
#PBS -j oe
#PBS -m abe
#PBS -N dgemv

cd $PBS_O_WORKDIR

orcc -v dgemv3.ancc.c > dgemv3.ancc.rs.data

#matlab -nodesktop -nosplash -nojvm -r "randomsearchd;quit;" 

