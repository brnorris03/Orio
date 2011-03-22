#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

#include "decl.h"

double rtclock()
{
  struct timezone tzp;
  struct timeval tp;
  int stat;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}

int main()
{
  init_input_vars();

  double annot_t_start=0, annot_t_end=0, annot_t_total=0;
  int annot_i;

  for (annot_i=0; annot_i<REPS; annot_i++)
    {
      annot_t_start = rtclock();

#ifdef DYNAMIC
      {
	int i,j;

	double* t7 = (double*) malloc(n*sizeof(double));
	double* t8 = (double*) malloc(n*sizeof(double));
	double* t3 = (double*) malloc(n*sizeof(double));
	for (i = 0; i < n; i+=1) {
	  double* t17 = B + i * n;
	  double t32 = 0;
	  for (j = 0; j < n; j+=1) {
	    double t31 = x[j];
	    double t30 = t17[j];
	    t32 += (t30*t31);
	  }
	  t7[i] = t32;
	}
	for (i = 0; i < n; i+=1) {
	  double t20 = t7[i];
	  t8[i] = (b*t20);
	}
	for (i = 0; i < n; i+=1) {
	  double* t11 = A + i * n;
	  double t29 = 0;
	  for (j = 0; j < n; j+=1) {
	    double t28 = x[j];
	    double t27 = t11[j];
	    t29 += (t27*t28);
	  }
	  t3[i] = t29;
	}
	for (i = 0; i < n; i+=1) {
	  double t14 = t3[i];
	  double t24 = t8[i];
	  y[i] = (t24+(a*t14));
	}
      }
#else
      {	
	int i,j;
	
	double* t7 = (double*) malloc(n*sizeof(double));
	double* t8 = (double*) malloc(n*sizeof(double));
	double* t3 = (double*) malloc(n*sizeof(double));
	
	for (i = 0; i < n; i+=1) {
	  double t32 = 0;
	  for (j = 0; j < n; j+=1) {
	    t32 += B[i][j]*x[j];
	  }
	  t7[i] = t32;
	}
	
	for (i = 0; i < n; i+=1) {
	  t8[i] = b*t7[i];
	}
	
	for (i = 0; i < n; i+=1) {
	  double t29 = 0;
	  for (j = 0; j < n; j+=1) {
	    t29 += A[i][j]*x[j];
	  }
	  t3[i] = t29;
	}
	for (i = 0; i < n; i+=1) {
	  y[i] = t8[i] + a*t3[i];
	}
      }
#endif

      annot_t_end = rtclock();
      annot_t_total += annot_t_end - annot_t_start;
    }

  annot_t_total = annot_t_total / REPS;

#ifndef TEST
  printf("%f\n", annot_t_total);
#else
  {
    int i, j;
    for (i=0; i<n; i++) {
      if (i%100==0)
        printf("\n");
      printf("%f ",y[i]);
    }
  }
#endif

  return ((int) y[0]);

}

