#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N bicgkernel.src2.c

cd $PBS_O_WORKDIR

orcc -v bicgkernel.src2.c > bicgkernel.src2.c.rs.data

