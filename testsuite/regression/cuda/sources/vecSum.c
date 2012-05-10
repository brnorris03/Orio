void VecSum(int n, double *x, double s) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for (i=0; i<=n-1; i++)
    s=s+x[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    s=s+x[i];

  /*@ end @*/
}
