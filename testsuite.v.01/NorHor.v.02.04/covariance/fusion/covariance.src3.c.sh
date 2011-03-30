#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N covariance.src3.c

cd $PBS_O_WORKDIR

orcc -v covariance.src3.c > covariance.src3.c.rs.data

