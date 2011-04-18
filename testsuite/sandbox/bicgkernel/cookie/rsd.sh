#!/bin/sh
#PBS -l nodes=1
#PBS -l walltime=100:00:00
#PBS -j oe
#PBS -m abe
#PBS -N bicgkernel

cd $PBS_O_WORKDIR

orcc -v bicgkernel.src2.c > bicgkernel.src2.rs.data



