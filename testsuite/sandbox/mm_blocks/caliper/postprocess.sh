#!/bin/bash

#mv caliper.json caliper$1-"$2".log
caliperdata="caliper.log"
echo "=======================================================================================" >> $caliperdata
echo "$1: $2" >> $caliperdata
cat caliper.txt >>  $caliperdata
rm -f caliper.txt
