void MatScale_SeqDIA(double* A, double a) {

  register int i;

  /*@ begin Loop(transform OpenCL(workGroups=32, workItemsPerGroup=32, cacheBlocks=True, vecHint=4, sizeHint=True, unrollInner=4, clFlags='-cl-fast-relaxed-math')

  for(i=0;i<=nz-1;i++) {
    A[i] *= a;
  }

  ) @*/

  for(i=0;i<=nz-1;i++) {
    A[i] *= a;
  }

  /*@ end @*/
}
