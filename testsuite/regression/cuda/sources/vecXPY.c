void VecXPY(int n, double *x, double *y) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=1024, maxBlocks=65535)
        for (i=0; i<=n-1; i++)
          y[i]=x[i]+y[i];
    ) @*/

    for (i=0; i<=n-1; i++)
        y[i]=x[i]+y[i];

    /*@ end @*/
}
