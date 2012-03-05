void VecWAXPY(int n, double *w, double a, double *x, double *y) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=16, cacheBlocks=True, pinHostMem=False, streamCount=2)
        for (i=0; i<=n-1; i++)
          w[i]=a*x[i]+y[i];
    ) @*/

    for (i=0; i<=n-1; i++)
        w[i]=a*x[i]+y[i];

    /*@ end @*/
}
