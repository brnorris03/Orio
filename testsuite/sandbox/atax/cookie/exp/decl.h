
#define NX N
#define NY N

#ifdef DYNAMIC
int nx;
int ny;
double *A;
double *x;
double *y;
#else
int nx;
int ny;
double A[NX][NY];
double x[NY];
double y[NY];
#endif


void init_input_vars() {
  int i, j;
#ifdef DYNAMIC
  A = (double*) malloc((NX*NY)*sizeof(double));
  x = (double*) malloc((NY)*sizeof(double));
  y = (double*) malloc((NY)*sizeof(double));
#endif
  nx = NX;
  ny = NY;
  for (i=0; i<=NY-1; i++) {
    x[i]=(i+1)/NY/4.0;
    y[i]=(i+1)/NY/5.0;
  }
  for (i=0; i<=NX-1; i++) {
    for (j=0; j<=NY-1; j++) {
#ifdef DYNAMIC
      A[i*NY+j]=(i*j)/NX/2.0;
#else
      A[i][j]=(i*j)/NX/2.0;
#endif
    }
  }
}

