void VecXPY(int n, double *x, double *y) {

    register int i;
    int lb = 4;

    /*@ begin Loop (
          transform CUDA(threadCount=16, cacheBlocks=True, pinHostMem=False, streamCount=2)
        for (i=lb; i<=n-1; i++)
          y[i]=x[i]+y[i];
    ) @*/

    for (i=lb; i<=n-1; i++)
        y[i]=x[i]+y[i];

    /*@ end @*/
}
