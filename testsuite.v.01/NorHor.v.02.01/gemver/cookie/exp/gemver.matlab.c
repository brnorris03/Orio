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
	double* t14 = (double*) malloc(n*sizeof(double));
	for (i = 0; i < n; i+=1) {
	  double t31 = u2[i];
	  double t24 = u1[i];
	  double* t27 = A + i * n;
	  double* t37 = B + i * n;
	  for (j = 0; j < n; j+=1) {
	    double t61 = v2[j];
	    double t54 = v1[j];
	    double t57 = t27[j];
	    t37[j] = ((t57+(t24*t54))+(t31*t61));
	  }
	}
	for (i = 0; i < n; ++i)
	  t14[i] = 0.0;
	for (i = 0; i < n; i+=1) {
	  double t39 = y[i];
	  double* t38 = B + i * n;
	  for (j = 0; j < n; j+=1) {
	    double t68 = t38[j];
	    t14[j] += (t39*t68);
	  }
	}
	for (i = 0; i < n; i+=1) {
	  double t41 = t14[i];
	  double t45 = z[i];
	  x[i] = (t45+(b*t41));
	}
	for (i = 0; i < n; i+=1) {
	  double* t48 = B + i * n;
	  double t73 = 0;
	  for (j = 0; j < n; j+=1) {
	    double t72 = x[j];
	    double t71 = t48[j];
	    t73 += (t71*t72);
	  }
	  w[i] = (a*t73);
	}
      }
#else
      {
	int i,j;
	double* tmp0 = (double*) malloc(n*sizeof(double));
	double tmp1;
	
	for (i = 0; i <= n-1; i+=1) {
	  for (j = 0; j <= n-1; j+=1) {
	    B[i][j] = A[i][j] + u1[i]*v1[j] + u2[i]*v2[j];
	  }
	}
	
	for (i = 0; i <= n-1; i+=1)
	  tmp0[i] = 0.0;
	
	for (i = 0; i <= n-1; i+=1) {
	  for (j = 0; j <= n-1; j+=1) {
	    tmp0[j] = tmp0[j] + y[i]*B[i][j];
	  }
	}
	
	for (i = 0; i <= n-1; i+=1) {
	  x[i] = z[i] + b*tmp0[i];
	}
	
	for (i = 0; i <= n-1; i+=1) {
	  tmp1 = 0;
	  for (j = 0; j <= n-1; j+=1) {
	    tmp1 = tmp1 + B[i][j]*x[j];
	  }
	  w[i] = a*tmp1;
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
      printf("%f ",w[i]);
    }
  }
#endif

  return ((int) w[0]);

}
