void VecDot(int n, double *x, double *y, double r) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for (i=0; i<=n-1; i++)
    r+=x[i]*y[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    r+=x[i]*y[i];

  /*@ end @*/
}
