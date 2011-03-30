#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N dgemv.ancc.src4.c

cd $PBS_O_WORKDIR

orcc -v dgemv.ancc.src4.c > dgemv.ancc.src4.c.rs.data

