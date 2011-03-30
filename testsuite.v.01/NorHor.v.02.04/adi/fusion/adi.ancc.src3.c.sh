#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N adi.ancc.src3.c

cd $PBS_O_WORKDIR

orcc -v adi.ancc.src3.c > adi.ancc.src3.c.rs.data

