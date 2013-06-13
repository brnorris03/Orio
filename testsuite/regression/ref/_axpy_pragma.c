void axpy(int n, double *y, double a, double *x) {
    register int i;
#pragma Orio Loop(transform Unroll(ufactor=3, parallelize=True))
{
  int i;
#pragma omp parallel for private(i)
  for (i=0; i<=n-3; i=i+3) {
    y[i]=y[i]+a*x[i];
    y[(i+1)]=y[(i+1)]+a*x[(i+1)];
    y[(i+2)]=y[(i+2)]+a*x[(i+2)];
  }
  for (i=n-((n-(0))%3); i<=n-1; i=i+1) 
    y[i]=y[i]+a*x[i];
}
#pragma Oiro
}
