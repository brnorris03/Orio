void VecScaleMult(int n, double a, double *x) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=1024, maxBlocks=65535)
        for (i=0; i<=n-1; i++)
          x[i]=a*x[i];
    ) @*/

    for (i=0; i<=n-1; i++)
        x[i]=a*x[i];

    /*@ end @*/
}
