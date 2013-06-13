void axpy(int n, double *y, double a, double *x) {
  register int i;

  /*@ begin Loop(transform Unroll(ufactor=5, parallelize=False)

  for (i=0; i<=n-1; i++)
    y[i]+=a*x[i];

  ) @*/
  {
    int i;
    for (i=0; i<=n-5; i=i+5) {
      y[i]=y[i]+a*x[i];
      y[(i+1)]=y[(i+1)]+a*x[(i+1)];
      y[(i+2)]=y[(i+2)]+a*x[(i+2)];
      y[(i+3)]=y[(i+3)]+a*x[(i+3)];
      y[(i+4)]=y[(i+4)]+a*x[(i+4)];
    }
    for (i=n-((n-(0))%5); i<=n-1; i=i+1) 
      y[i]=y[i]+a*x[i];
  }
  /*@ end @*/
}
