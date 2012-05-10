void VecAXPY(int n, double a, double *x, double *y) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for (i=0; i<=n-1; i++)
    y[i]+=a*x[i];

  ) @*/

  for (i=0; i<=n-1; i++)
      y[i]=a*x[i]+y[i];

  /*@ end @*/
}
