#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N lu.src3.c

cd $PBS_O_WORKDIR

orcc -v lu.src3.c > lu.src3.c.rs.data

