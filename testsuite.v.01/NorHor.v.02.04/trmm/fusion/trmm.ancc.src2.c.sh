#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N trmm.ancc.src2.c

cd $PBS_O_WORKDIR

orcc -v trmm.ancc.src2.c > trmm.ancc.src2.c.rs.data

