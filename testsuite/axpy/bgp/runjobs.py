#!/usr/bin/env python

import os, sys, time, random, re

def runExp(ename, flag, src_file, libs, numthreads):

    print '-=-=-=-=-=-=-=-=-=-=-=-=-'
    print ename
    print '-=-=-=-=-=-=-=-=-=-=-=-=-'

    CC = 'bgxlc_r -O3 -qstrict -qarch=450d -qtune=450 -qhot' 

    REPS = [100000,10000,10000,1000,1000,500,500,100,50,20]
    N = [10,100,1000,10000,50000,100000,500000,1000000,5000000,10000000]
    T = [5,5,5,5,5,10,10,10,15,15]

    # iterate over each problem size
    jobs_queue = []
    for reps, n, t in zip(REPS,N,T):

        # generate a random integer
        i = random.randint(1,100000)

        # build the code
        build_cmd = ('%s %s -DN=%s -DREPS=%s -o %s%s %s %s' %
                     (CC, flag, n, reps, ename, i, src_file, libs))
        print '********************************'
        print build_cmd
        print '********************************'
        os.system(build_cmd)
        
        # submit a job
        batch_cmd = ('qsub -n 1 -t %s -q short --env "OMP_NUM_THREADS=%s" ./%s%s' %
                     (t, numthreads, ename, i))
        print '********************************'
        print batch_cmd
        print '********************************'
        f = os.popen(batch_cmd)
        output = f.read()
        f.close()
            
        # get the job ID
        job_id = output.strip().split('\n')[-1]
        print job_id

        # insert job into the queue
        jobs_queue.append([job_id, reps, n, t])

    # iterate over each job stored in the queue
    for job_id, reps, n, t in jobs_queue:
        
        # wait till the job is done
        while 1:
            print '.',
            time.sleep(10)
            status_cmd = 'qstat | grep %s | wc -l' % job_id
            f = os.popen(status_cmd)
            status = f.read().strip()
            f.close()
            if status == '0':
                break
        print ''

    # iterate over each job stored in the queue
    for job_id, reps, n, t in jobs_queue:

        # read the performance numbers
        runtime = -1
        mflops = -1
        out_file = '%s.output' % job_id
        f = open(out_file)
        results = f.read().strip().split()
        f.close()
        if results and len(results) == 2:
            runtime, mflops = results
            
        # store the performance results
        perf_numbers = '%s.%s\t%s\t%s\n' % (ename, n, mflops, runtime)
        f = open('results.txt', 'a')
        f.write(perf_numbers)
        f.close()
        
        # remove all unnecessary files
        rm_cmd = 'rm %s.*' % job_id
        print '********************************'
        print rm_cmd
        print '********************************'
        os.system(rm_cmd)
        

# Name, Flags, Source files, LIBS flag, Number of threads

runExp('base_seq', '-qsmp=noauto', 'axpy4.base.c', '', 1)
runExp('base_par', '-qsmp=auto', 'axpy4.base.c', '', 4)


runExp('essl_seq', '-qsmp=noauto', 'axpy4.blas.c',
       '-L/bgsys/ibm_essl/sles10/prod/opt/ibmmath/lib -lesslbg ' +
       '-L/opt/ibmcmp/xlf/bg/11.1/bglib -lxlf90_r -lxlfmath -lm',
       1)
runExp('essl_par', '-qsmp=auto', 'axpy4.blas.c',
       '-L/bgsys/ibm_essl/sles10/prod/opt/ibmmath/lib -lesslsmpbg ' +
       '-L/opt/ibmcmp/xlf/bg/11.1/bglib -lxlf90_r -lxlfmath -lm',
       4)


runExp('goto_seq', '-qsmp=noauto', 'axpy4.blas.c',
       '-L/soft/apps/LIBGOTO -lgoto -L/opt/ibmcmp/xlf/bg/11.1/bglib -lxlf90_r ' +
       '-lpthread -lxlfmath -lm',
       1)
runExp('goto_par', '-qsmp=auto', 'axpy4.blas.c',
       '-L/soft/apps/LIBGOTO -lgoto -L/opt/ibmcmp/xlf/bg/11.1/bglib -lxlf90_r ' +
       '-lpthread -lxlfmath -lm',
       4)


runExp('orio_seq', '-qsmp=omp:noauto -DORIO_SEQ', 'axpy4.orio.c', '', 1)
runExp('orio_par', '-qsmp=omp:noauto -DORIO_PAR', 'axpy4.orio.c', '', 4)


