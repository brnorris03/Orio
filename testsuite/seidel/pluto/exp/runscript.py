#!/usr/bin/env python

import os, sys, re


def runExp(nthreadss, cc, src_file, out_file, flags, libs):
    compile_cmd = '%s %s -o %s %s %s' % (cc, flags, out_file, src_file, libs)
    print '*************************************'
    print compile_cmd
    print '*************************************'
    os.system(compile_cmd)

    rtimes = []
    for nt in nthreadss:
        run_cmd = 'export OMP_NUM_THREADS=%s; ./%s' % (nt, out_file)
        print '*************************************'
        print run_cmd
        print '*************************************'
        f = os.popen(run_cmd)
        output = f.read()
        f.close()
        rtime = eval(output)
        rtimes.append(rtime)
    return rtimes

def countFlops(T, N, rtimes):
    ops = 9*T*(N-2)*(N-2)
    mflopss = []
    for rtime in rtimes:
        mflops = 1.0*ops/(rtime*1000000)
        mflopss.append(mflops)
    return mflopss

def myDiff(fname1, fname2):
    f1 = open(fname1)
    c1 = f1.read()
    f1.close()
    f2 = open(fname2)
    c2 = f2.read()
    f2.close()
    cls1 = c1.split('\n')
    cls2 = c2.split('\n')
    maxlength=100
    diffs = []
    total_diffs = 0
    total_nums = 0
    for l1,l2 in zip(cls1, cls2):
        l1 = l1.split()
        l2 = l2.split()
        for n1,n2 in zip(l1,l2):
            total_nums+=1
            n1=eval(n1)
            n2=eval(n2)
            d=n1-n2
            if not (-1<d<1):
                total_diffs+=1
                if len(diffs) == 0:
                    diffs.append(d)
                    continue
                maxdiff=diffs[-1]
                if len(diffs) < maxlength:
                    diffs.append(d)
                    diffs.sort()
                elif d > maxdiff:
                    diffs.pop(0)
                    diffs.append(d)
                    diffs.sort()
    return (total_diffs==0, diffs, total_diffs, total_nums)

def checkCorrectness(optflag = '-O0'):
    T=1
    N=512
    compile_cmd = ('icc -O0 -DREPS=1 -DT=%s -DN=%s -DTEST -o base_test seidel.base.c -lm' % 
                   (T, N))
    run_cmd = 'export OMP_NUM_THREADS=1; ./base_test'
    print '***********************'
    print compile_cmd
    print run_cmd
    print '***********************'
    os.system(compile_cmd)
    f = os.popen(run_cmd)
    output = f.read()
    f.close()
    f = open('output_base', 'w')
    f.write(output)
    f.close()
    
    fnames = [
        'seidel.base.pluto.seq.c', 
        'seidel.base.pluto.par.c', 
        'seidel.pluto_orio.seq_par.c',
        ]
    for fname in fnames:
        compile_cmd = (('icc %s -openmp -DREPS=1 -DT=%s -DN=%s -DTEST ' +
                        '-o opt_test %s -lm') % (optflag, T, N, fname))
        run_cmd = 'export OMP_NUM_THREADS=1; ./opt_test'
        print '***********************'
        print compile_cmd
        print run_cmd
        print '***********************'
        os.system(compile_cmd)
        f = os.popen(run_cmd)
        output = f.read()
        f.close()
        f = open(('output_%s.out' % fname), 'w')
        f.write(output)
        f.close()
        print '*************************************'
        print '.... comparing results to the base'
        print '*************************************'
        is_correct, diffs, total_diffs, total_nums = myDiff('output_base', ('output_%s.out' % fname))
        if not is_correct:
            percent=(1.0*total_diffs)/total_nums
            print 'error: -----------INCORRECT RESULTS-----------'
            print '---------- total numbers=%s---------' % total_nums
            print '---------- total different numbers=%s (%s)---------' % (total_diffs, percent)
            print '----------- inaccuracy differences -----------'
            for d in diffs:
                print ' %s ' % d,
            print '\n'
        else:
            print '-------PASSED CORRECTNESS CHECKING--------'

# correctness checking
OPTFLAG = '-O3'
checkCorrectness()
checkCorrectness(OPTFLAG)

# parallel case
if 0:
    reps = 1
    T = 1024
    N = 1024
    flags = '-DREPS=%s -DT=%s -DN=%s' % (reps, T, N)

    rtimes_base_par = runExp([1,2,3,4], 'icc %s -parallel' % OPTFLAG, 
                             'seidel.base.c', 'base_par', flags, '-lm')
    mflopss_base_par = countFlops(T,N,rtimes_base_par)
    
    rtimes_pluto_par = runExp([1,2,3,4], 'icc %s -openmp' % OPTFLAG, 
                              'seidel.base.pluto.par.c', 'pluto_par', flags, '-lm')
    mflopss_pluto_par = countFlops(T,N,rtimes_pluto_par)
    
    rtimes_orio_par = runExp([1,2,3,4], 'icc %s -openmp' % OPTFLAG, 
                             'seidel.pluto_orio.seq_par.c', 'orio_par', flags, '-lm')
    mflopss_orio_par = countFlops(T,N,rtimes_orio_par)
    
    print mflopss_base_par
    print mflopss_pluto_par
    print mflopss_orio_par
    
# sequential case
if 1:

    reps = 1
    T = 1024

    mflopss_base = []
    mflopss_pluto = []
    mflopss_orio = []
    for N in [256,512,1024,2048,4096]:
        flags = '-DREPS=%s -DT=%s -DN=%s' % (reps, T, N)
        rtimes_base = runExp([1], 'icc %s' % OPTFLAG, 
                             'seidel.base.c', 'base_seq', flags, '-lm')
        p = countFlops(T,N,rtimes_base)
        mflopss_base.append(p[0])
        rtimes_pluto = runExp([1], 'icc %s' % OPTFLAG, 
                              'seidel.base.pluto.seq.c', 'pluto_seq', flags, '-lm')
        p = countFlops(T,N,rtimes_pluto)
        mflopss_pluto.append(p[0])
        rtimes_orio = runExp([1], 'icc %s -openmp' % OPTFLAG, 
                             'seidel.pluto_orio.seq_par.c', 'orio_seq', flags, '-lm')
        p = countFlops(T,N,rtimes_orio)
        mflopss_orio.append(p[0])
        
    print mflopss_base
    print mflopss_pluto
    print mflopss_orio
    
