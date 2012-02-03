void VecAXPBY(int n, double a, double *x, double b, double *y) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=1024, maxBlocks=65535)
        for (i=0; i<=n-1; i++)
          y[i]=a*x[i]+b*y[i];
    ) @*/

    for (i=0; i<=n-1; i++)
        y[i]=a*x[i]+b*y[i];

    /*@ end @*/
}
