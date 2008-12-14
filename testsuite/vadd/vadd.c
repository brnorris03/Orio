
void vadd(int n, double *x, double *w, double *y, double *z) {

    register int i;

/*@ begin Align (x1[],x2[],x3[],x4[],y[]) @*/
/*@ begin Loop (
  transform Unroll(ufactor=20, parallelize=True)
    for (i=0; i<=n-1; i++)
      x[i]=w[i]+y[i]+z[i];
      ) @*/

    for (i=0; i<=n-1; i++)
        x[i]=w[i]+y[i]+z[i];

/*@ end @*/
/*@ end @*/

}

