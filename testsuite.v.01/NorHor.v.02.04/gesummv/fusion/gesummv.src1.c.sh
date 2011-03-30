#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N gesummv.src1.c

cd $PBS_O_WORKDIR

orcc -v gesummv.src1.c > gesummv.src1.c.rs.data

