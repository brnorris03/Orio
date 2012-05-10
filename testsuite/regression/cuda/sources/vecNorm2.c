void VecNorm2(int n, double *x, double r) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for (i=0; i<=n-1; i++)
    r+=x[i]*x[i];
  r=sqrt(r);

  ) @*/

  for (i=0; i<=n-1; i++)
    r=r+x[i]*x[i];
  r=sqrt(r);

  /*@ end @*/
}
