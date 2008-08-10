double *x1;
double *x2;
double *x3;
double *x4;
double *x5;
double *y;
double a1;
double a2;
double a3;
double a4;
double a5;

void malloc_arrays() {
    int i1;
    x1 = (double*) malloc((N) * sizeof(double));
    x2 = (double*) malloc((N) * sizeof(double));
    x3 = (double*) malloc((N) * sizeof(double));
    x4 = (double*) malloc((N) * sizeof(double));
    x5 = (double*) malloc((N) * sizeof(double));
    y = (double*) malloc((N) * sizeof(double));
}

void init_input_vars() {
    int i1;
    for (i1=0; i1<N; i1++) {
        x1[i1] = (i1+1) % 4 + 1;
        x2[i1] = (i1+5) % 10 + 1;
        x3[i1] = (i1+3) % 6 + 1;
        x4[i1] = (i1+9) % 9 + 1;
        x5[i1] = (i1+8) % 15 + 1;
        y[i1] = 0;
    }
    a1 = (double) 6;
    a2 = (double) 7;
    a3 = (double) 4;
    a4 = (double) 1;
    a5 = (double) 9;
}
