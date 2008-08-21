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
	int i,j,it,jt;
	for (i=0; i<=n-1; i=i+1) {
	  x[i]=0;
	  w[i]=0;
	}
	for (i=0; i<=n-1; i=i+1) {
	  double* tA=A+i*n;
	  double* tB=B+i*n;
	  for (j=0; j<=n-1; j=j+1) {
	    tB[j]=u2[i]*v2[j]+u1[i]*v1[j]+tA[j];
	    x[j]=y[i]*tB[j]+x[j];
	  }
	}
	for (i=0; i<=n-1; i=i+1) {
	  x[i]=b*x[i]+z[i];
	}
	{
	  register int ub=n-1;
#pragma omp parallel for
	  for (i=0; i<=ub; i=i+1) {
	    double tmp1=w[i];
	    double* tB=B+i*n;
	    for (j=0; j<=n-1; j=j+1) {
	      tmp1=tmp1+tB[j]*x[j];
	    }
	    w[i]=a*tmp1;
	  }
	}
      }
#else
      {
	int i,j,it,jt;
	for (i=0; i<=n-1; i=i+1) {
	  x[i]=0;
	  w[i]=0;
	}
	for (j=0; j<=n-1; j=j+1) {
	  for (i=0; i<=n-1; i=i+1) {
	    B[j][i]=u2[j]*v2[i]+u1[j]*v1[i]+A[j][i];
	    x[i]=y[j]*B[j][i]+x[i];
	  }
	}
	for (i=0; i<=n-1; i=i+1) {
	  x[i]=b*x[i]+z[i];
	}
	{
#pragma omp parallel for
	  for (i=0; i<=n-1; i=i+1) {
	    double scv_1;
	    scv_1=w[i];
	    for (j=0; j<=n-1; j=j+1) {
	      scv_1=scv_1+B[i][j]*x[j];
	    }
	    scv_1=a*scv_1;
	    w[i]=scv_1;
	  }
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
