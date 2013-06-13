void axpy(int n, double *y, double a, double *x) {
    register int i;
#pragma Orio Loop(transform Unroll(ufactor=3, parallelize=True))
    for (i=0; i<=n-1; i++) y[i] += a * x[i];
#pragma Oiro
}
