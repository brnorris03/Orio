void VecSum(int n, double *x, double s) {

    register int i;
    int s = 0; // identity for summation

    /*@ begin Loop (
          transform CUDA(threadCount=1024, maxBlocks=65535)
        for (i=0; i<=n-1; i++)
          s=s+x[i];
    ) @*/

    for (i=0; i<=n-1; i++)
        s=s+x[i];

    /*@ end @*/
}
