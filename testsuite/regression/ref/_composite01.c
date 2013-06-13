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
  {
    int i;
    register int cbv_1;
    cbv_1=n-2;
#pragma ivdep
#pragma vector always
    for (i=0; i<=cbv_1; i=i+2) {
      y[i]=y[i]+a*x[i];
      y[(i+1)]=y[(i+1)]+a*x[(i+1)];
    }
    register int cbv_2, cbv_3;
    cbv_2=n-((n-(0))%2);
    cbv_3=n-1;
#pragma ivdep
#pragma vector always
    for (i=cbv_2; i<=cbv_3; i=i+1) 
      y[i]=y[i]+a*x[i];
  }
  /*@ end @*/
}
