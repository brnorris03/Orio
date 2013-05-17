
void axpy4(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4) {

    register int i;

/*@ begin Align (x1[],x2[],x3[],x4[],y[]) @*/
/*@ begin Loop (
  transform Unroll(ufactor=20, parallelize=True)
    for (i=0; i<=n-1; i++)
      y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i];
      ) @*/

    for (i=0; i<=n-1; i++)
        y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i];

/*@ end @*/
/*@ end @*/

}

