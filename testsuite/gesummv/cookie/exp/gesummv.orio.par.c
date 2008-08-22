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
#pragma omp parallel for private(i,j)
	for (i=0; i<=n-1; i=i+1) {
	  double tmp=0;
	  y[i]=0;
	  {
	    double* tA=A+i*n;
	    double* tB=B+i*n;
	    register int cbv_1;
	    cbv_1=n-8;
#pragma ivdep
#pragma vector always
	    for (j=0; j<=cbv_1; j=j+8) {
	      tmp=tA[j]*x[j]+tmp;
	      tmp=tA[(j+1)]*x[(j+1)]+tmp;
	      tmp=tA[(j+2)]*x[(j+2)]+tmp;
	      tmp=tA[(j+3)]*x[(j+3)]+tmp;
	      tmp=tA[(j+4)]*x[(j+4)]+tmp;
	      tmp=tA[(j+5)]*x[(j+5)]+tmp;
	      tmp=tA[(j+6)]*x[(j+6)]+tmp;
	      tmp=tA[(j+7)]*x[(j+7)]+tmp;
	      y[i]=tB[j]*x[j]+y[i];
	      y[i]=tB[(j+1)]*x[(j+1)]+y[i];
	      y[i]=tB[(j+2)]*x[(j+2)]+y[i];
	      y[i]=tB[(j+3)]*x[(j+3)]+y[i];
	      y[i]=tB[(j+4)]*x[(j+4)]+y[i];
	      y[i]=tB[(j+5)]*x[(j+5)]+y[i];
	      y[i]=tB[(j+6)]*x[(j+6)]+y[i];
	      y[i]=tB[(j+7)]*x[(j+7)]+y[i];
	    }
	    register int cbv_2;
	    cbv_2=n-1;
#pragma ivdep
#pragma vector always
	    for (; j<=cbv_2; j=j+1) {
	      tmp=tA[j]*x[j]+tmp;
	      y[i]=tB[j]*x[j]+y[i];
	    }
	  }
	  y[i]=a*tmp+b*y[i];
	}
      }
#else
      {
	int i,j;
#pragma omp parallel for private(i,j)
	for (i=0; i<=n-1; i=i+1) {
	  double tmp=0;
	  y[i]=0;
	  {
	    register int cbv_1;
	    cbv_1=n-8;
#pragma ivdep
#pragma vector always
	    for (j=0; j<=cbv_1; j=j+8) {
	      tmp=A[i][j]*x[j]+tmp;
	      tmp=A[i][(j+1)]*x[(j+1)]+tmp;
	      tmp=A[i][(j+2)]*x[(j+2)]+tmp;
	      tmp=A[i][(j+3)]*x[(j+3)]+tmp;
	      tmp=A[i][(j+4)]*x[(j+4)]+tmp;
	      tmp=A[i][(j+5)]*x[(j+5)]+tmp;
	      tmp=A[i][(j+6)]*x[(j+6)]+tmp;
	      tmp=A[i][(j+7)]*x[(j+7)]+tmp;
	      y[i]=B[i][j]*x[j]+y[i];
	      y[i]=B[i][(j+1)]*x[(j+1)]+y[i];
	      y[i]=B[i][(j+2)]*x[(j+2)]+y[i];
	      y[i]=B[i][(j+3)]*x[(j+3)]+y[i];
	      y[i]=B[i][(j+4)]*x[(j+4)]+y[i];
	      y[i]=B[i][(j+5)]*x[(j+5)]+y[i];
	      y[i]=B[i][(j+6)]*x[(j+6)]+y[i];
	      y[i]=B[i][(j+7)]*x[(j+7)]+y[i];
	    }
	    register int cbv_2;
	    cbv_2=n-1;
#pragma ivdep
#pragma vector always
	    for (; j<=cbv_2; j=j+1) {
	      tmp=A[i][j]*x[j]+tmp;
	      y[i]=B[i][j]*x[j]+y[i];
	    }
	  }
	  y[i]=a*tmp+b*y[i];
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

