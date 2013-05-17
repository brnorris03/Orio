void axpy(int n, double *y, double a, double *x) {
  register int i;

  /*@ begin Loop(
        transform Composite(
          unrolljam = (['i'],[2]),
          vector    = (True, ['ivdep','vector always'])
        )

  for (i=0; i<=n-1; i++)
    y[i]+=a*x[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    y[i]+=a*x[i];

  /*@ end @*/
}
