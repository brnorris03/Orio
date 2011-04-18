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
      int i,j;
      for (i = 0; i < ny; ++i)
	s[i] = 0.0;
      for (i = 0; i < nx; i+=1) {
	double t12 = r[i];
	double* t8 = A + i * ny;
	double t16 = 0;
	for (j = 0; j < ny; j+=1) {
	  double t15 = p[j];
	  double t14 = t8[j];
	  s[j] += (t12*t14);
	  t16 += (t14*t15);
	}
	q[i] = t16;
      }
#else
      int i,j;
      for (i = 0; i < ny; ++i)
	s[i] = 0.0;
      
      for (i = 0; i < nx; i+=1) {
	double tmp = 0;
	for (j = 0; j < ny; j+=1) {
	  s[j] = s[j] + r[i]*A[i][j];
	  tmp =  tmp + A[i][j]*p[j];
	}
	q[i] = tmp;
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
      printf("%f ",s[i]);
    }
    printf("\n");
    for (i=0; i<nx; i++) {
      if (i%100==0)
        printf("\n");
      printf("%f ",q[i]);
    }
  }
#endif

  return ((int) (s[0]+q[0]));

}

