#!/bin/bash
# Expects two command-line arguments:
#   $1: the name of the performance test executable, something like __orio_perftest2.exe
#   $2: The string representation of the parameter coordinate indices, e.g., "[2, 0, 4]"

#mv caliper.json caliper$1-"$2".log
caliperdata="caliper.log"
echo "=======================================================================================" >> $caliperdata
echo "$1: $2" >> $caliperdata
cat caliper.txt >>  $caliperdata
rm -f caliper.txt
