void VecScaleMult(int n, double a, double *x) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for (i=0; i<=n-1; i++)
    x[i]*=a;

  ) @*/

  for (i=0; i<=n-1; i++)
    x[i]=a*x[i];

  /*@ end @*/
}
