#!/bin/sh
#PBS -l nodes=1
#PBS -l walltime=100:00:00
#PBS -j oe
#PBS -m abe
#PBS -N fdtd

cd $PBS_O_WORKDIR

orcc -v fdtd-2d.src2.c > fdtd-2d.src2.rs.data



