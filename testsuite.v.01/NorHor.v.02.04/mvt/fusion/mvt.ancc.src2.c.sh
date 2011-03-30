#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N mvt.ancc.src2.c

cd $PBS_O_WORKDIR

orcc -v mvt.ancc.src2.c > mvt.ancc.src2.c.rs.data

