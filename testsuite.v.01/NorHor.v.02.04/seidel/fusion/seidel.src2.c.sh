#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N seidel.src2.c

cd $PBS_O_WORKDIR

orcc -v seidel.src2.c > seidel.src2.c.rs.data

