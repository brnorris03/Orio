void MatScale_SeqDIA(double* A, double a) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for(i=0;i<=nz-1;i++) {
    A[i] *= a;
  }

  ) @*/

  for(i=0;i<=nz-1;i++) {
    A[i] *= a;
  }

  /*@ end @*/
}
