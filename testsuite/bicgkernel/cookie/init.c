
#define NX N
#define NY N

#ifdef DYNAMIC
int nx;
int ny;
double *A;
double *r;
double *s;
double *p;
double *q;
#else
int nx;
int ny;
double A[NX][NY];
double r[NX];
double s[NY];
double p[NY];
double q[NX];
#endif


void init_input_vars() {
  int i, j;
#ifdef DYNAMIC
  A = (double*) malloc((NX*NY)*sizeof(double));
  r = (double*) malloc((NX)*sizeof(double));
  s = (double*) malloc((NY)*sizeof(double));
  p = (double*) malloc((NY)*sizeof(double));
  q = (double*) malloc((NX)*sizeof(double));
#endif
  nx = NX;
  ny = NY;
  for (i=0; i<=NX-1; i++) {
    r[i]=(i+1)/NX/2.0;
    q[i]=(i+1)/NX/3.0;
  }
  for (i=0; i<=NY-1; i++) {
    s[i]=(i+1)/NY/4.0;
    p[i]=(i+1)/NY/5.0;
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

