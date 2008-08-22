
#ifdef DYNAMIC
double a;
double b;
int n;
double *A;
double *B;
double *x;
double *y;
#else
double a;
double b;
int n;
double A[N][N];
double B[N][N];
double x[N];
double y[N];
#endif


void init_input_vars() {
  int i, j;
#ifdef DYNAMIC
  A = (double*) malloc((N*N)*sizeof(double));
  B = (double*) malloc((N*N)*sizeof(double));
  x = (double*) malloc((N)*sizeof(double));
  y = (double*) malloc((N)*sizeof(double));
#endif
  a = 1.5;
  b = 2.5;
  n = N;
  for (i=0; i<=N-1; i++) {
    x[i]=(i+1)/N/3.0;
    y[i]=(i+1)/N/4.0;
    for (j=0; j<=N-1; j++) {
#ifdef DYNAMIC
      A[i*N+j]=(i*j)/N/2.0;
      B[i*N+j]=(i*j)/N/3.0;
#else
      A[i][j]=(i*j)/N/2.0;
      B[i][j]=(i*j)/N/3.0;
#endif
    }
  }
}

