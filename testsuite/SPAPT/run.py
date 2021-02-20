#!/usr/bin/env python3

# Usage: run.py [output_dir]
# If output directory is specified, it's used, otherwise a timestamp-named directory is created

from orio.main.util.globals import *
import sys
import glob

kernels = ['adi', 'atax', 'bicgkernel', 'correlation', 'covariance', 'dgemv3', 'fdtd',
           'gemver', 'gesummv', 'hessian', 'jacobi', 'lu', 'mm', 'mvt', 'seidel',
           'stencil3d', 'tensor-contraction', 'trmm']

outdir = timestamp()
spaptdir = os.path.abspath(os.getcwd())
orcc = os.path.abspath(os.path.join(spaptdir.replace('testsuite/SPAPT',''),'orcc'))

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    outdir = sys.argv[1]

if not os.path.exists(outdir):
    os.mkdir(outdir)
    for kernel in kernels:
        print('rsync -ak %s ./"%s"' % (kernel,outdir))
        os.system('rsync -ak %s ./"%s"' % (kernel,outdir))
if not os.path.isabs(outdir):
    wdir = os.path.join(spaptdir,outdir)
else:
    wdir = outdir

print("========== Starting all tests in SPAPT (%s) =========" % timestamp())
for kernel in kernels:
    # Go into results directory for each kernel
    os.chdir(os.path.join(wdir,kernel))
    #os.system('pwd & ls -l')
    print('==== %s ====' % kernel)
    # Run Orio
    for variant in range(1,7):
        input = glob.glob('*.src%d.c' % variant)[0]
        orio_cmd = orcc + ' -vzk ' + input
        print(orio_cmd)
        #os.system(orio_cmd)

print("========== Successfully ran all tests in SPAPT (%s) =========" % timestamp())
