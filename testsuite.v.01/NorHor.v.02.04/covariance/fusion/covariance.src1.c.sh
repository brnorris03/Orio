#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N covariance.src1.c

cd $PBS_O_WORKDIR

orcc -v covariance.src1.c > covariance.src1.c.rs.data

