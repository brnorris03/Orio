void VecNorm2(int n, double *x, double r) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, cacheBlocks=True, streamCount=2)

  for (i=0; i<=n-1; i++)
    r=r+x[i]*x[i];
  r=sqrt(r);

  ) @*/

  for (i=0; i<=n-1; i++)
    r=r+x[i]*x[i];
  r=sqrt(r);

  /*@ end @*/
}
