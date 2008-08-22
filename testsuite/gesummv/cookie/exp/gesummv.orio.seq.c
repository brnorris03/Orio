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
	for (i=0; i<=n-3; i=i+3) {
	  double scv_2, scv_3, scv_4, scv_5, scv_6, scv_7;
	  scv_3=0;
	  scv_2=0;
	  scv_7=0;
	  scv_5=0;
	  scv_6=0;
	  scv_4=0;
	  double* tA0=A+i*n;
	  double* tA1=A+(i+1)*n;
	  double* tA2=A+(i+2)*n;
	  double* tB0=B+i*n;
	  double* tB1=B+(i+1)*n;
	  double* tB2=B+(i+2)*n;
	  register int cbv_1;
	  cbv_1=n-1;
#pragma ivdep
#pragma vector always
	  for (j=0; j<=cbv_1; j=j+1) {
	    double scv_1;
	    scv_1=x[j];
	    scv_3=tA0[j]*scv_1+scv_3;
	    scv_2=tA1[j]*scv_1+scv_2;
	    scv_7=tA2[j]*scv_1+scv_7;
	    scv_5=tB0[j]*scv_1+scv_5;
	    scv_6=tB1[j]*scv_1+scv_6;
	    scv_4=tB2[j]*scv_1+scv_4;
	  }
	  scv_5=a*scv_3+b*scv_5;
	  scv_6=a*scv_2+b*scv_6;
	  scv_4=a*scv_7+b*scv_4;
	  y[i]=scv_5;
	  y[(i+1)]=scv_6;
	  y[(i+2)]=scv_4;
	}
	for (; i<=n-1; i=i+1) {
	  double scv_9, scv_10;
	  scv_10=0;
	  scv_9=0;
	  double* tA0=A+i*n;
	  double* tB0=B+i*n;	  
	  register int cbv_2;
	  cbv_2=n-1;
#pragma ivdep
#pragma vector always
	  for (j=0; j<=cbv_2; j=j+1) {
	    double scv_8;
	    scv_8=x[j];
	    scv_10=tA0[j]*scv_8+scv_10;
	    scv_9=tB0[j]*scv_8+scv_9;
	    }
	  scv_9=a*scv_10+b*scv_9;
	  y[i]=scv_9;
	}
      }
#else      
      {
	int i,j;
	for (i=0; i<=n-3; i=i+3) {
	  double scv_2, scv_3, scv_4, scv_5, scv_6, scv_7;
	  scv_3=0;
	  scv_2=0;
	  scv_7=0;
	  scv_5=0;
	  scv_6=0;
	  scv_4=0;
	  register int cbv_1;
	  cbv_1=n-1;
#pragma ivdep
#pragma vector always
	  for (j=0; j<=cbv_1; j=j+1) {
	    double scv_1;
	    scv_1=x[j];
	    scv_3=A[i][j]*scv_1+scv_3;
	    scv_2=A[(i+1)][j]*scv_1+scv_2;
	    scv_7=A[(i+2)][j]*scv_1+scv_7;
	    scv_5=B[i][j]*scv_1+scv_5;
	    scv_6=B[(i+1)][j]*scv_1+scv_6;
	    scv_4=B[(i+2)][j]*scv_1+scv_4;
	  }
	  scv_5=a*scv_3+b*scv_5;
	  scv_6=a*scv_2+b*scv_6;
	  scv_4=a*scv_7+b*scv_4;
	  y[i]=scv_5;
	  y[(i+1)]=scv_6;
	  y[(i+2)]=scv_4;
	}
	for (; i<=n-1; i=i+1) {
	  double scv_9, scv_10;
	  scv_10=0;
	  scv_9=0;
	  register int cbv_2;
	  cbv_2=n-1;
#pragma ivdep
#pragma vector always
	  for (j=0; j<=cbv_2; j=j+1) {
	    double scv_8;
	    scv_8=x[j];
	    scv_10=A[i][j]*scv_8+scv_10;
	    scv_9=B[i][j]*scv_8+scv_9;
	    }
	  scv_9=a*scv_10+b*scv_9;
	  y[i]=scv_9;
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

