
CWD=`pwd`

for file in $(find  -name "*src4*.sh")
do
dn="${file%%${file##*/}}"
cd $dn
job=$(ls *src4*.sh)
echo $job
qsub $job
sleep 1
cd $CWD
done
