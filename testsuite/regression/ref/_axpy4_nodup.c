void axpy4(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4) {

    int i;

/*@ begin Loop(transform Unroll(ufactor=3, parallelize=True)) @*/
{
  int i;
#pragma omp parallel for private(i)
  for (i=0; i<=n-3; i=i+3) {
    y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i];
    y[(i+1)]=y[(i+1)]+a1*x1[(i+1)]+a2*x2[(i+1)]+a3*x3[(i+1)]+a4*x4[(i+1)];
    y[(i+2)]=y[(i+2)]+a1*x1[(i+2)]+a2*x2[(i+2)]+a3*x3[(i+2)]+a4*x4[(i+2)];
  }
  for (i=n-((n-(0))%3); i<=n-1; i=i+1) 
    y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i];
}
/*@ end @*/

}

