#!/usr/bin/env python

import os, sys, re

def printFloats(ls):
    s=''
    s+='['
    for i,f in enumerate(ls):
        if i: s+=', '
        s+='%2.3f' % f
    s+=']'
    print s

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

def countFlops(N, rtimes):
    ops = N*(N-1)/2 + 2*N*(N-1)*(2*N-1)/6
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
            try:
                n1=eval(n1)
                n2=eval(n2)
            except:
                print 'error: cannot evaluate'
                print 'n1=%s' % n1
                print 'n2=%s' % n2
                sys.exit(1)
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
    N=500
    compile_cmd = 'gcc -O0 -DREPS=1 -DN=%s -DTEST -o base_test lu.base.c -lm' % (N)
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
        'lu.pluto.seq.l1tile.c', 
        'lu.pluto.seq.l2tile.c', 
        'lu.pluto.par.l1tile.c', 
        'lu.pluto.par.l2tile.c', 
        'lu.orio.seq.small.c',
        'lu.orio.seq.large.c',
        'lu.orio.par.c',
        ]
    for fname in fnames:
        compile_cmd = (('icc %s -openmp -DREPS=1 -DN=%s -DTEST -o opt_test %s -lm') % 
                       (optflag, N, fname))
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
if 1:
    reps = 1
    N = 4000
    flags = '-DREPS=%s -DN=%s' % (reps, N)

    mflopss_base = []
    mflopss_pluto_l1 = []
    mflopss_pluto_l2 = []
    mflopss_orio = []

    rtimes_base = runExp([1,2,3,4,5,6,7,8], 'icc %s -parallel' % OPTFLAG, 
                         'lu.base.c', 'base_par', flags, '-lm')
    mflopss_base = countFlops(N,rtimes_base)
    
    rtimes_pluto_l1 = runExp([1,2,3,4,5,6,7,8], 'icc %s -openmp' % OPTFLAG, 
                             'lu.pluto.par.l1tile.c', 'pluto_par', flags, '-lm')
    mflopss_pluto_l1 = countFlops(N,rtimes_pluto_l1)
    
    rtimes_pluto_l2 = runExp([1,2,3,4,5,6,7,8], 'icc %s -openmp' % OPTFLAG, 
                             'lu.pluto.par.l2tile.c', 'pluto_par', flags, '-lm')
    mflopss_pluto_l2 = countFlops(N,rtimes_pluto_l2)
    
    rtimes_orio = runExp([1,2,3,4,5,6,7,8], 'icc %s -openmp' % OPTFLAG, 
                         'lu.orio.par.c', 'orio_par', flags, '-lm')
    mflopss_orio = countFlops(N,rtimes_orio)
    
    printFloats(mflopss_base)
    printFloats(mflopss_pluto_l1)
    printFloats(mflopss_pluto_l2)
    printFloats(mflopss_orio)
    
# sequential case
if 1:
    reps = 1

    mflopss_base = []
    mflopss_pluto_l1 = []
    mflopss_pluto_l2 = []
    mflopss_orio_small = []
    mflopss_orio_large = []

    for N in [125,250,500,1000,2000,4000,8000]:
        flags = '-DREPS=%s -DN=%s' % (reps, N)
        
        rtimes_base = runExp([1], 'icc %s' % OPTFLAG, 
                             'lu.base.c', 'base_seq', flags, '-lm')
        p = countFlops(N,rtimes_base)
        mflopss_base.append(p[0])
        
        rtimes_pluto_l1 = runExp([1], 'icc %s' % OPTFLAG, 
                                 'lu.pluto.seq.l1tile.c', 'pluto_seq', flags, '-lm')
        p = countFlops(N,rtimes_pluto_l1)
        mflopss_pluto_l1.append(p[0])
        
        rtimes_pluto_l2 = runExp([1], 'icc %s' % OPTFLAG, 
                                 'lu.pluto.seq.l2tile.c', 'pluto_seq', flags, '-lm')
        p = countFlops(N,rtimes_pluto_l2)
        mflopss_pluto_l2.append(p[0])
        
        rtimes_orio_small = runExp([1], 'icc %s -openmp' % OPTFLAG, 
                                   'lu.orio.seq.small.c', 'orio_seq', flags, '-lm')
        p = countFlops(N,rtimes_orio_small)
        mflopss_orio_small.append(p[0])

        rtimes_orio_large = runExp([1], 'icc %s -openmp' % OPTFLAG, 
                                   'lu.orio.seq.large.c', 'orio_seq', flags, '-lm')
        p = countFlops(N,rtimes_orio_large)
        mflopss_orio_large.append(p[0])
        
    printFloats(mflopss_base)
    printFloats(mflopss_pluto_l1)
    printFloats(mflopss_pluto_l2)
    printFloats(mflopss_orio_small)
    printFloats(mflopss_orio_large)
    
