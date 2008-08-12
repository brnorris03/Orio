
void axpy4(int* _n, double *y, double* _a1, double *x1, double* _a2, double *x2,
           double* _a3, double *x3, double* _a4, double *x4) {
    int n = *_n;
    double a1 = *_a1;
    double a2 = *_a2;
    double a3 = *_a3;
    double a4 = *_a4;
    int i;
#pragma omp parallel for
    for (i=0; i<=n-1; i++)
        y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
}

