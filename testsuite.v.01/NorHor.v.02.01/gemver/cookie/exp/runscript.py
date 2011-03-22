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

def runExp(nthreadss, cc, src_file, flags, libs):
    compile_cmd = '%s %s %s %s' % (cc, flags, src_file, libs)
    print '*************************************'
    print compile_cmd
    print '*************************************'
    os.system(compile_cmd)

    rtimes = []
    for nt in nthreadss:
        run_cmd = 'export OMP_NUM_THREADS=%s; ./a.out' % (nt)
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
    ops = (8*N*N+3*N)
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

def checkCorrectness(optflag, arrtype):
    N=5000
    compile_cmd = (('gcc -DDYNAMIC -O0 -DREPS=1 -DN=%s -DTEST gemver.matlab.c -lm') % N)
    run_cmd = 'export OMP_NUM_THREADS=1; ./a.out'
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
        'gemver.matlab.c', 
        'gemver.orio.seq.c', 
        'gemver.orio.par.c', 
        ]
    for fname in fnames:
        compile_cmd = (('icc %s %s -openmp -DREPS=1 -DN=%s -DTEST %s -lm') % 
                       (arrtype, optflag, N, fname))
        run_cmd = 'export OMP_NUM_THREADS=1; ./a.out'
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
if 1:
    checkCorrectness('-O0', '-DDYNAMIC')
    checkCorrectness(OPTFLAG, '-DDYNAMIC')
    checkCorrectness('-O0', '')
    checkCorrectness(OPTFLAG, '')

# parallel case
if 0:
    reps = 10
    N = 10000
    #N = 20000
    flags = '-DREPS=%s -DN=%s' % (reps, N)

    rtimes_matlab_static =[]
    rtimes_matlab_dynamic =[]
    rtimes_orio_static =[]
    rtimes_orio_dynamic =[]

    mflopss_matlab_static = []
    mflopss_matlab_dynamic = []
    mflopss_orio_static = []
    mflopss_orio_dynamic = []

    if 0 and N <= 10000:
        rtimes = runExp([1,2,3,4,5,6,7,8], 'icc %s -parallel' % OPTFLAG, 
                        'gemver.matlab.c', flags, '-lm')
        rtimes_matlab_static = rtimes
        mflopss_matlab_static = countFlops(N,rtimes)

    for N in range(2000,22000,2000):
        #rtimes = runExp([1,2,3,4,5,6,7,8], 'icc %s -DDYNAMIC -parallel' % OPTFLAG,
        flags = '-DREPS=%s -DN=%s' % (reps, N)
        rtimes = runExp([8], 'icc %s -DDYNAMIC -parallel' % OPTFLAG,
                    'gemver.matlab.c', flags, '-lm')
        rtimes_matlab_dynamic.extend(rtimes)
        mflopss_matlab_dynamic.extend(countFlops(N,rtimes))

    if 0 and N <= 10000:
        rtimes = runExp([1,2,3,4,5,6,7,8], 'icc %s -openmp' % OPTFLAG, 
                        'gemver.orio.par.c', flags, '-lm')
        rtimes_orio_static = rtimes
        mflopss_orio_static = countFlops(N,rtimes)
        
    for N in range(2000,22000,2000):
        #rtimes = runExp([1,2,3,4,5,6,7,8], 'icc %s -DDYNAMIC -openmp' % OPTFLAG,
        flags = '-DREPS=%s -DN=%s' % (reps, N)
        rtimes = runExp([8], 'icc %s -DDYNAMIC -openmp' % OPTFLAG,
                    'gemver.orio.par.c', flags, '-lm')
        rtimes_orio_dynamic.extend(rtimes)
        mflopss_orio_dynamic.extend(countFlops(N,rtimes))

    print '--- Parallel: seconds (static arrays) ---'
    print 'matlab=',
    printFloats(rtimes_matlab_static)
    print 'orio=',
    printFloats(rtimes_orio_static)

    print '--- Parallel: Mflops/sec (static arrays) ---'
    print 'matlab=',
    printFloats(mflopss_matlab_static)
    print 'orio=',
    printFloats(mflopss_orio_static)

    print '--- Parallel: seconds (dynamic arrays) ---'
    print 'matlab=',
    printFloats(rtimes_matlab_dynamic)
    print 'orio=',
    printFloats(rtimes_orio_dynamic)

    print '--- Parallel: Mflops/sec (dynamic arrays) ---'
    print 'matlab=',
    printFloats(mflopss_matlab_dynamic)
    print 'orio=',
    printFloats(mflopss_orio_dynamic)

    
# sequential case
if 1:

    rtimes_matlab_static =[]
    rtimes_matlab_dynamic =[]
    rtimes_orio_static =[]
    rtimes_orio_dynamic =[]

    mflopss_matlab_static = []
    mflopss_matlab_dynamic = []
    mflopss_orio_static = []
    mflopss_orio_dynamic = []

    for N in [2000,4000,6000,8000,10000]:
    #for N in [12000,14000,16000,18000,20000]:

        if N < 5000:
            reps = 200
        elif N < 8000:
            reps = 100
        elif N < 11000:
            reps = 5
        else:
            reps = 2

        flags = '-DREPS=%s -DN=%s' % (reps, N)
        
        if N <= 10000:
            rtimes = runExp([1], 'icc %s' % OPTFLAG, 'gemver.matlab.c', flags, '-lm')
            p = countFlops(N,rtimes)
            rtimes_matlab_static.append(rtimes[0])
            mflopss_matlab_static.append(p[0])
        
        rtimes = runExp([1], 'icc %s -DDYNAMIC' % OPTFLAG, 'gemver.matlab.c', flags, '-lm')
        p = countFlops(N,rtimes)
        rtimes_matlab_dynamic.append(rtimes[0])
        mflopss_matlab_dynamic.append(p[0])
        
        if N <= 10000:
            rtimes = runExp([1], 'icc %s' % OPTFLAG, 'gemver.orio.seq.c', flags, '-lm')
            p = countFlops(N,rtimes)
            rtimes_orio_static.append(rtimes[0])
            mflopss_orio_static.append(p[0])
        
        rtimes = runExp([1], 'icc %s -DDYNAMIC' % OPTFLAG, 'gemver.orio.seq.c', flags, '-lm')
        p = countFlops(N,rtimes)
        rtimes_orio_dynamic.append(rtimes[0])
        mflopss_orio_dynamic.append(p[0])

    print '--- Sequential: seconds (static arrays) ---'
    print 'matlab=',
    printFloats(rtimes_matlab_static)
    print 'orio=',
    printFloats(rtimes_orio_static)

    print '--- Sequential: Mflops/sec (static arrays) ---'
    print 'matlab=',
    printFloats(mflopss_matlab_static)
    print 'orio=',
    printFloats(mflopss_orio_static)

    print '--- Sequential: seconds (dynamic arrays) ---'
    print 'matlab=',
    printFloats(rtimes_matlab_dynamic)
    print 'orio=',
    printFloats(rtimes_orio_dynamic)

    print '--- Sequential: Mflops/sec (dynamic arrays) ---'
    print 'matlab=',
    printFloats(mflopss_matlab_dynamic)
    print 'orio=',
    printFloats(mflopss_orio_dynamic)

