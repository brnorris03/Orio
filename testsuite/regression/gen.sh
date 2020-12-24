#!/bin/bash
# cleanup
/bin/rm -rf src/_*.c src/tests/*

# Loop optimizations
for i in `ls src/*.c`; do 
	dir=$(dirname $i)
	name=$(basename ${i%%.*})
	sed -e "s|@FILE@|$name|" $dir/template.py > "$dir/tests/test_${name}.py" ; 
done

# TODO: OpenCL, etc.

