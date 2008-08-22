
#ifdef DYNAMIC
double a;
double b;
int n;
double *A;
double *B;
double *u1;
double *u2;
double *v1;
double *v2;
double *y;
double *z;
double *w;
double *x;
#else
double a;
double b;
int n;
double A[N][N];
double B[N][N];
double u1[N];
double u2[N];
double v1[N];
double v2[N];
double y[N];
double z[N];
double w[N];
double x[N];
#endif


void init_input_vars() {
  int i, j;
#ifdef DYNAMIC
  A = (double*) malloc((N*N)*sizeof(double));
  B = (double*) malloc((N*N)*sizeof(double));
  u1 = (double*) malloc((N)*sizeof(double));
  u2 = (double*) malloc((N)*sizeof(double));
  v1 = (double*) malloc((N)*sizeof(double));
  v2 = (double*) malloc((N)*sizeof(double));
  y = (double*) malloc((N)*sizeof(double));
  z = (double*) malloc((N)*sizeof(double));
  w = (double*) malloc((N)*sizeof(double));
  x = (double*) malloc((N)*sizeof(double));
#endif
  a = 1.5;
  b = 2.5;
  n = N;
  for (i=0; i<=N-1; i++) {
    u1[i]=(i+1)/N/1.0;
    u2[i]=(i+1)/N/2.0;
    v1[i]=(i+1)/N/3.0;
    v2[i]=(i+1)/N/4.0;
    y[i]=(i+1)/N/5.0;
    z[i]=(i+1)/N/6.0;
    w[i]=(i+1)/N/7.0;
    x[i]=(i+1)/N/8.0;
    for (j=0; j<=N-1; j++) {
#ifdef DYNAMIC
      A[i*N+j]=(i*j)/N;
      B[i*N+j]=0;
#else
      A[i][j]=(i*j)/N;
      B[i][j]=0;
#endif
    }
  }
}

