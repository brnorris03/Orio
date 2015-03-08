#!/bin/sh

#orcuda -v --keep-temps --post-command="python ~/repos/cubin_parse/cubinProcessing.py -v --json --outdir $PWD/outdir/data%iter --sass ./%exe" vecAXPY.c 
#orcuda -v --keep-temps --meta -o $PWD/matVec2D-64/data%iter --post-command="python ~/repos/cubin_parse/cubinProcessing.py -v --json --outdir $PWD/matVec2D-64/data%iter --all ./%exe 50" matVec2D-64.c 

mkdir matVec2D-32
orcuda -v --keep-temps --meta -o $PWD/matVec2D-32/data%iter --post-command="python ~/repos/cubin_parse/cubinProcessing.py -v --json --outdir $PWD/matVec2D-32/data%iter --events 'TIME,CUDA.Tesla_K40c.domain_d.inst_executed,CUDA.Tesla_K40c.domain_d.active_cycles,CUDA.Tesla_K40c.domain_d.l1_local_load_miss,CUDA.Tesla_K40c.domain_d.l1_global_load_miss' --all ./%exe 50" matVec2D-32.c

mkdir sourcefiles
mv __* sourcefiles
mv sourcefiles matVec2D-32
mv *.log matVec2D-32

##                                                                                                                                                                                                       
mkdir matVec2D-64
orcuda -v --keep-temps --meta -o $PWD/matVec2D-64/data%iter --post-command="python ~/repos/cubin_parse/cubinProcessing.py -v --json --outdir $PWD/matVec2D-64/data%iter --events 'TIME,CUDA.Tesla_K40c.domain_d.inst_executed,CUDA.Tesla_K40c.domain_d.active_cycles,CUDA.Tesla_K40c.domain_d.l1_local_load_miss,CUDA.Tesla_K40c.domain_d.l1_global_load_miss' --all ./%exe 50" matVec2D-64.c

mkdir sourcefiles
mv __* sourcefiles
mv sourcefiles matVec2D-64
mv *.log matVec2D-64

##                                                                                                                                                                                                       
mkdir matVec2D-128
orcuda -v --keep-temps --meta -o $PWD/matVec2D-128/data%iter --post-command="python ~/repos/cubin_parse/cubinProcessing.py -v --json --outdir $PWD/matVec2D-128/data%iter --events 'TIME,CUDA.Tesla_K40c.domain_d.inst_executed,CUDA.Tesla_K40c.domain_d.active_cycles,CUDA.Tesla_K40c.domain_d.l1_local_load_miss,CUDA.Tesla_K40c.domain_d.l1_global_load_miss' --all ./%exe 50" matVec2D-128.c

mkdir sourcefiles
mv __* sourcefiles
mv sourcefiles matVec2D-128
mv *.log matVec2D-128

##                                                                                                                                                                                                       
mkdir matVec2D-256
orcuda -v --keep-temps --meta -o $PWD/matVec2D-256/data%iter --post-command="python ~/repos/cubin_parse/cubinProcessing.py -v --json --outdir $PWD/matVec2D-256/data%iter --events 'TIME,CUDA.Tesla_K40c.domain_d.inst_executed,CUDA.Tesla_K40c.domain_d.active_cycles,CUDA.Tesla_K40c.domain_d.l1_local_load_miss,CUDA.Tesla_K40c.domain_d.l1_global_load_miss' --all ./%exe 50" matVec2D-256.c

mkdir sourcefiles
mv __* sourcefiles
mv sourcefiles matVec2D-256
mv *.log matVec2D-256

##                                                                                                                                                                                                       
mkdir matVec2D-512
orcuda -v --keep-temps --meta -o $PWD/matVec2D-512/data%iter --post-command="python ~/repos/cubin_parse/cubinProcessing.py -v --json --outdir $PWD/matVec2D-512/data%iter --events 'TIME,CUDA.Tesla_K40c.domain_d.inst_executed,CUDA.Tesla_K40c.domain_d.active_cycles,CUDA.Tesla_K40c.domain_d.l1_local_load_miss,CUDA.Tesla_K40c.domain_d.l1_global_load_miss' --all ./%exe 50" matVec2D-512.c

mkdir sourcefiles
mv __* sourcefiles
mv sourcefiles matVec2D-512
mv *.log matVec2D-512
