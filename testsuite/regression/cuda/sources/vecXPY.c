void VecXPY(int n, double *x, double *y) {

  register int i;
  int lb = 4;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for (i=lb; i<=n-1; i++)
    y[i]+=x[i];

  ) @*/

  for (i=lb; i<=n-1; i++)
    y[i]+=x[i];

  /*@ end @*/
}
