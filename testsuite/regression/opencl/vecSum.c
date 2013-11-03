void VecSum(int n, double *x, double s) {

  register int i;

  /*@ begin Loop(transform OpenCL(workItemsPerGroup=16, workGroups=32, vecHint=4, sizeHint=True)

  for (i=0; i<=n-1; i++)
    s=s+x[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    s=s+x[i];

  /*@ end @*/
}
