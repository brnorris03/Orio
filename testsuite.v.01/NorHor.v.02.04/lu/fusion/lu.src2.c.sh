#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N lu.src2.c

cd $PBS_O_WORKDIR

orcc -v lu.src2.c > lu.src2.c.rs.data

