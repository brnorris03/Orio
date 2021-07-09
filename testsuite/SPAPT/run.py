#!/usr/bin/env python3

# Usage: run.py [output_dir]
# If output directory is specified, it's used, otherwise a timestamp-named directory is created


gcc_arch = 'znver2'  # skylake-avx512
gcc_arch = 'skylake-avx512'
icc_arch = 'skylake'

kernels = ['adi', 'atax', 'bicgkernel', 'correlation', 'covariance', 'dgemv3', 'fdtd',
           'gemver', 'gesummv', 'hessian', 'jacobi', 'lu', 'mm', 'mvt', 'seidel',
           'stencil3d', 'tensor-contraction', 'trmm']
# kernels = ['axpy4','axpy4a'] # used only for testing

search_methods = [('Randomsearch',5000), ('Randomlocal',5000)]
    #, 'Mlsearch', 'msimplex', 'simplex', 'firefly', 'direct']

reps = {1 : 10, 2 : 7, 3 : 5, 4 : 10, 5 : 7, 6 : 5 }

import os, sys, glob, socket, datetime, massedit, threading

# include Orio's source directory in the Python's search path, assume we are running
# this script in top_dir/testsuite/SPAPT
top_dir = os.path.dirname(os.path.realpath(os.path.join('..','..',__file__,'orio')))
sys.path.insert(0, os.path.dirname(top_dir))

import orio.main.orio_main


def timestamp():
    hostname = socket.gethostname()
    timestamp = hostname + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return timestamp

def setup(version, reps, search, maxruns):
    sources = glob.glob('*.src%d.c' % version)
    filenames = [x for x in sources if not x.startswith('_')]
    massedit.edit_files(filenames, ["re.sub(r'gcc ', 'gcc -Ofast -march=%s -mavx2 -m64 ', line)" % gcc_arch], dry_run=False)
    #massedit.edit_files(filenames, ["re.sub(r'gcc -O3 -fopenmp ','icc -O3 -mtune=%s -xCORE-AVX512 -qopt-zmm-usage=high -qopenmp ' % icc_arch, line)"], dry_run=False)
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
            kernel_dir = os.path.join(wdir,search,kernel)
            os.chdir(kernel_dir)
            #os.system('pwd & ls -l')
            print('==== %s, %s ====' % (kernel,search))
            # Run Orio
            for variant in range(1,7):
                setup(variant, reps[variant], search, maxruns) 
                input_file = glob.glob('*.src%d.c' % variant)[0]
                orio_cmd = orcc + ' -vk ' + input_file
                print(orio_cmd)
                import orio.main.orio_main
                if not dry_run:
                    archivedir = ('archive-%s' % input_file)[:-2] # strip the *.c suffix from backup dir
                    if os.path.exists(os.path.join(archivedir,'_'+input_file)): # this case has already been autotuned
                        print('\n<<=====>> Already autotuned, results in %s/%s\n' % (kernel_dir,archivedir))
                        continue
                    # dispatch to Orio's main
                    #orio.main.orio_main.start(['orcc','-vk',input_file], orio.main.orio_main.C_CPP)
                    x = threading.Thread(target=orio.main.orio_main.start, args=(['orcc','-vk',input_file], orio.main.orio_main.C_CPP))
                    x.start()
                    x.join()   # timeout not specified
                    archivedir = ('archive-%s' % input_file)[:-2] # strip the *.c suffix from backup dir
                    if not os.path.exists(archivedir): os.mkdir(archivedir)
                    print('mv _*.c tuning*.log %s' % archivedir)
                    os.system('mv _*.c tuning*.log %s' % archivedir)
                    os.system('cp %s %s' % (input_file,archivedir))
                    os.system('rm -f _*.exe')
   
        print("========== Successfully ran all tests in SPAPT (%s) =========" % timestamp())


if __name__ == "__main__":
    run(dry_run = False)

