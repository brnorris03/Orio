
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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
  for (i1=0; i1<N; i1++)
   x1[i1] = (i1) % 5 + 1;
  for (i1=0; i1<N; i1++)
   x2[i1] = (i1) % 5 + 1;
  for (i1=0; i1<N; i1++)
   x3[i1] = (i1) % 5 + 1;
  for (i1=0; i1<N; i1++)
   x4[i1] = (i1) % 5 + 1;
  for (i1=0; i1<N; i1++)
   x5[i1] = (i1) % 5 + 1;
  for (i1=0; i1<N; i1++)
   y[i1] = 0;
  a1 = (double) 6.99846222671;
  a2 = (double) 7.61751115547;
  a3 = (double) 4.56538602829;
  a4 = (double) 1.74370739872;
  a5 = (double) 9.31495181566;
}

double getClock()
{
  struct timezone tzp;
  struct timeval tp;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}

int main(int argc, char *argv[])
{
    malloc_arrays();
    init_input_vars();

    int one = 1;
    int n = N;

    double orio_t_start, orio_t_end, orio_t_total=0;
    int orio_i;

    orio_t_start = getClock(); 
    for (orio_i=0; orio_i<REPS; orio_i++)
    {

	daxpy(&n, &a1, x1, &one, y, &one);
	daxpy(&n, &a2, x2, &one, y, &one);
	daxpy(&n, &a3, x3, &one, y, &one);
	daxpy(&n, &a4, x4, &one, y, &one);
	daxpy(&n, &a5, x5, &one, y, &one);
	
    }
    orio_t_end = getClock();
    orio_t_total = orio_t_end - orio_t_start;

    orio_t_total = orio_t_total / REPS; 
    double mflops = (10.0*N)/(orio_t_total*1000000);

    printf("%.6f\t%.3f\n", orio_t_total, mflops);

    return y[0];
}

