#!/usr/bin/env python3

# Usage: run.py [output_dir]
# If output directory is specified, it's used, otherwise a timestamp-named directory is created

kernels = ['adi', 'atax', 'bicgkernel', 'correlation', 'covariance', 'dgemv3', 'fdtd',
           'gemver', 'gesummv', 'hessian', 'jacobi', 'lu', 'mm', 'mvt', 'seidel',
           'stencil3d', 'tensor-contraction', 'trmm']

search_methods = [('Randomsearch',10000), ('Randomlocal',10000), ('Randomsimple',10000)]
    #, 'Mlsearch', 'msimplex', 'simplex', 'firefly', 'direct']

reps = {1 : 20, 2 : 10, 3 : 5, 4 : 20, 5 : 10, 6 : 5 }

import os, sys, glob, socket, datetime, massedit

# include Orio's source directory in the Python's search path, assume we are running
# this script in top_dir/testsuite/SPAPT
top_dir = os.path.dirname(os.path.realpath(os.path.join('..','..',__file__,'orio')))
sys.path.insert(0, os.path.dirname(top_dir))



def timestamp():
    hostname = socket.gethostname()
    timestamp = hostname + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return timestamp

def setup(version, reps, search, maxruns):
    filenames = glob.glob('*.src%d.c' % version)
    print(filenames)
    massedit.edit_files(filenames, ["re.sub(r'gcc ', 'gcc-10 -Ofast -mavx2 -m64 ', line)"], dry_run=False)
    #massedit.edit_files(filenames, ["re.sub(r'gcc -O3 -fopenmp ','icc -O3 -mtune=skylake -xCORE-AVX512 -qopt-zmm-usage=high -qopenmp ', line)"], dry_run=False)
    massedit.edit_files(filenames, ["re.sub(r'arg repetitions = 35;', 'arg repetitions = %d;', line)" % reps], dry_run=False)
    massedit.edit_files(filenames, ["re.sub(r\"arg algorithm = 'Randomsearch';\", \"arg algorithm = '%s';\", line)" % search], dry_run=False)
    massedit.edit_files(filenames, ["re.sub(r'arg total_runs = 10000;', 'arg total_runs = %d;', line)" % maxruns], dry_run=False)

def run(dry_run=True):
    outdir = timestamp()
    spaptdir = os.path.abspath(os.getcwd())
    orcc = os.path.abspath(os.path.join(spaptdir.replace('testsuite/SPAPT',''),'orcc'))

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        outdir = sys.argv[1]
 
    if not os.path.exists(outdir): os.mkdir(outdir)
    for search, maxruns in search_methods:
        resdir = os.path.join(outdir,search)
        if not os.path.exists(resdir):
            os.mkdir(resdir)
            for kernel in kernels:
                print('rsync -ak %s ./"%s"' % (kernel,resdir))
                os.system('rsync -ak %s ./"%s"' % (kernel,resdir))

    wdir = outdir
    if not os.path.isabs(outdir):
        wdir = os.path.join(spaptdir,outdir)
  
    for search,maxruns in search_methods:
        print("Search method:", search)
        print("========== Starting all tests in SPAPT (%s) =========" % timestamp())
        for kernel in kernels:
            # Go into results directory for each kernel
            os.chdir(os.path.join(wdir,search,kernel))
            #os.system('pwd & ls -l')
            print('==== %s ====' % kernel)
            # Run Orio
            for variant in range(1,7):
                setup(variant, reps[variant], search, maxruns) 
                input = glob.glob('*.src%d.c' % variant)[0]
                orio_cmd = orcc + ' -vk ' + input
                print(orio_cmd)
                import orio.main.orio_main
                if not dry_run:
                    # dispatch to Orio's main
                    import orio.main.orio_main
                    orio.main.orio_main.start(['orcc','-vk',input], orio.main.orio_main.C_CPP)
   
        print("========== Successfully ran all tests in SPAPT (%s) =========" % timestamp())


if __name__ == "__main__":
    run(dry_run = False)

