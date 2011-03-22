

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MSIZE 10000
#define NSIZE 10000
int M;
int N;
double **a;
double *y_1;
double *y_2;
double *x1;
double *x2;
void malloc_arrays() {
  int i1,i2;
  a = (double**) malloc((M) * sizeof(double*));
  for (i1=0; i1<M; i1++) {
   a[i1] = (double*) malloc((N) * sizeof(double));
  }
  y_1 = (double*) malloc((N) * sizeof(double));
  y_2 = (double*) malloc((M) * sizeof(double));
  x1 = (double*) malloc((M) * sizeof(double));
  x2 = (double*) malloc((N) * sizeof(double));
}

void init_input_vars() {
  int i1,i2;
  M = MSIZE;
  N = NSIZE;
  for (i1=0; i1<M; i1++)
   for (i2=0; i2<N; i2++)
    a[i1][i2] = (i1+i2) % 5 + 1;
  for (i1=0; i1<N; i1++)
   y_1[i1] = (i1) % 5 + 1;
  for (i1=0; i1<M; i1++)
   y_2[i1] = (i1) % 5 + 1;
  for (i1=0; i1<M; i1++)
   x1[i1] = 0;
  for (i1=0; i1<N; i1++)
   x2[i1] = 0;
}



#ifdef BGP_COUNTER
#define SPRN_TBRL 0x10C // Time Base Read Lower Register (user & sup R/O)
#define SPRN_TBRU 0x10D // Time Base Read Upper Register (user & sup R/O)
#define _bgp_mfspr( SPRN )\
({\
  unsigned int tmp;\
  do {\
    asm volatile ("mfspr %0,%1" : "=&r" (tmp) : "i" (SPRN) : "memory" );\
  }\
  while(0);\
  tmp;\
})\

double getClock()
{
  union {
    unsigned int ul[2];
    unsigned long long ull;
  }
  hack;
  unsigned int utmp;
  do {
    utmp = _bgp_mfspr( SPRN_TBRU );
    hack.ul[1] = _bgp_mfspr( SPRN_TBRL );
    hack.ul[0] = _bgp_mfspr( SPRN_TBRU );
  }
  while(utmp != hack.ul[0]);
  return((double) hack.ull );
}
#else
double getClock()
{
  struct timezone tzp;
  struct timeval tp;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}
#endif

int main(int argc, char *argv[])
{
  malloc_arrays();
init_input_vars();


  double orio_t_start, orio_t_end, orio_t_total=0;
  int orio_i;

  for (orio_i=0; orio_i<REPS; orio_i++)
  {
    orio_t_start = getClock();
    
    


int i, j;
int ii, jj;
int iii, jjj;

/*@ begin Loop(
  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),(('ii','i'),T2_I,'iii'),(('jj','j'),T2_J,'jjj')],
    scalarreplace = (SCREP, 'double', 'scv_'),
    vector = (VEC, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,ii,jj,i,j)')
  )

for (i=0;i<=N-1;i++)
  for (j=0;j<=N-1;j++) 
  { 
    x1[i]=x1[i]+a[i][j]*y_1[j]; 
    x2[j]=x2[j]+a[i][j]*y_2[i]; 
  } 

) @*/
for (iii=0; iii<=N-1; iii=iii+256) 
  for (ii=iii; ii<=min(N-1,iii); ii=ii+256) 
    for (i=ii; i<=min(N-1,ii+255); i=i+1) 
      for (jjj=0; jjj<=N-1; jjj=jjj+256) 
        for (jj=jjj; jj<=min(N-1,jjj); jj=jj+256) 
          for (j=jj; j<=min(N-1,jj+255); j=j+1) {
            x1[i]=x1[i]+a[i][j]*y_1[j];
            x2[j]=x2[j]+a[i][j]*y_2[i];
          }
/*@ end @*/


    orio_t_end = getClock();
    orio_t_total += orio_t_end - orio_t_start;
    printf("try: %g\n", orio_t_end - orio_t_start);
  }
  orio_t_total = orio_t_total / REPS;
  
  printf("{'[0, 0, 0, 0, 0, 0, 0, 0, 0]' : %g}", orio_t_total);

  

  return 0;
}

