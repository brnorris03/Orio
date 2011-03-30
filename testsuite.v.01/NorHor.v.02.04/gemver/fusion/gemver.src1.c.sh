#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N gemver.src1.c

cd $PBS_O_WORKDIR

orcc -v gemver.src1.c > gemver.src1.c.rs.data

