#!/bin/sh
#PBS -l nodes=1
#PBS -l walltime=100:00:00
#PBS -j oe
#PBS -m abe
#PBS -N gemver

cd $PBS_O_WORKDIR

orcc -v gemver.src2.c > gemver.src2.rs.data



