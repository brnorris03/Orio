#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N .src*.c

cd $PBS_O_WORKDIR

orcc -v .src*.c > .src*.c.rs.data

