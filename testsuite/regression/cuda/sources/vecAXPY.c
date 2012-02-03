void VecAXPY(int n, double a, double *x, double *y) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=1024, maxBlocks=65535)
        for (i=0; i<=n-1; i++)
          y[i]=a*x[i]+y[i];
    ) @*/

    for (i=0; i<=n-1; i++)
        y[i]=a*x[i]+y[i];

    /*@ end @*/
}
