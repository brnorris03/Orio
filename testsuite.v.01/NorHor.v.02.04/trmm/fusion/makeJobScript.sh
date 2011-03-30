#!/bin/bash

for i in *.src*.c

do

echo $i

OUTFILE=$i.sh         # Name of the file to generate.
(
cat <<EOF
#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=300:00:00
#PBS -j oe
#PBS -m abe
#PBS -N $i

cd \$PBS_O_WORKDIR

orcc -v $i > $i.rs.data

EOF
) > $OUTFILE

chmod +x $OUTFILE

done

