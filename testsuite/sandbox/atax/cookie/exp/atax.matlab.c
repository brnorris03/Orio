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
	for (i = 0; i < ny; ++i)
	  y[i] = 0.0;
	for (i = 0; i < nx; i += 1) {
	  double *t6 = A + i * ny;
	  double t14 = 0;
    
	  for (j = 0; j < ny; j += 1) {
	    double t13 = x[j];
	    double t12 = t6[j];
	    t14 += (t12 * t13);
	  }
	  for (j = 0; j < ny; j += 1) {
	    double t15 = t6[j];
	    y[j] += (t15 * t14);
	  }
	}
      }


#else

      {
	int i,j;
	double tmp;

	for (i= 0; i<=ny-1; i++)
	  y[i] = 0.0;
	for (i = 0; i<=nx-1; i += 1) {
	  tmp = 0;
	  for (j = 0; j<=ny-1; j++) 
	    tmp = tmp + A[i][j]*x[j];
	  for (j = 0; j<=ny-1; j++) 
	    y[j] = y[j] + A[i][j]*tmp;
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
    for (i=0; i<ny; i++) {
      if (i%100==0)
        printf("\n");
      printf("%f ",y[i]);
    }
    printf("\n");
  }
#endif

  return ((int) y[0]);

}

