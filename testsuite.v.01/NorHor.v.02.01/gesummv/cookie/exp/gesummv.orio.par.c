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
  
{
#pragma omp parallel for private(j,i)
  for (i=0; i<=n-3; i=i+3) {
    double scv_6, scv_7, scv_8, scv_9, scv_10, scv_11;
    scv_6=y[(i+1)];
    scv_8=y[(i+2)];
    scv_9=y[i];
    scv_7=0;
    scv_10=0;
    scv_11=0;
    scv_9=0;
    scv_6=0;
    scv_8=0;
    for (j=0; j<=n-4; j=j+4) {
      double scv_1, scv_2, scv_3, scv_4;
      scv_1=x[j];
      scv_2=x[(j+3)];
      scv_3=x[(j+2)];
      scv_4=x[(j+1)];
      scv_7=A[i*n+j]*scv_1+scv_7;
      scv_10=A[(i+1)*n+j]*scv_1+scv_10;
      scv_11=A[(i+2)*n+j]*scv_1+scv_11;
      scv_7=A[i*n+j+1]*scv_4+scv_7;
      scv_10=A[(i+1)*n+j+1]*scv_4+scv_10;
      scv_11=A[(i+2)*n+j+1]*scv_4+scv_11;
      scv_7=A[i*n+j+2]*scv_3+scv_7;
      scv_10=A[(i+1)*n+j+2]*scv_3+scv_10;
      scv_11=A[(i+2)*n+j+2]*scv_3+scv_11;
      scv_7=A[i*n+j+3]*scv_2+scv_7;
      scv_10=A[(i+1)*n+j+3]*scv_2+scv_10;
      scv_11=A[(i+2)*n+j+3]*scv_2+scv_11;
      scv_9=B[i*n+j]*scv_1+scv_9;
      scv_6=B[(i+1)*n+j]*scv_1+scv_6;
      scv_8=B[(i+2)*n+j]*scv_1+scv_8;
      scv_9=B[i*n+j+1]*scv_4+scv_9;
      scv_6=B[(i+1)*n+j+1]*scv_4+scv_6;
      scv_8=B[(i+2)*n+j+1]*scv_4+scv_8;
      scv_9=B[i*n+j+2]*scv_3+scv_9;
      scv_6=B[(i+1)*n+j+2]*scv_3+scv_6;
      scv_8=B[(i+2)*n+j+2]*scv_3+scv_8;
      scv_9=B[i*n+j+3]*scv_2+scv_9;
      scv_6=B[(i+1)*n+j+3]*scv_2+scv_6;
      scv_8=B[(i+2)*n+j+3]*scv_2+scv_8;
    }
    for (; j<=n-1; j=j+1) {
      double scv_5;
      scv_5=x[j];
      scv_7=A[i*n+j]*scv_5+scv_7;
      scv_10=A[(i+1)*n+j]*scv_5+scv_10;
      scv_11=A[(i+2)*n+j]*scv_5+scv_11;
      scv_9=B[i*n+j]*scv_5+scv_9;
      scv_6=B[(i+1)*n+j]*scv_5+scv_6;
      scv_8=B[(i+2)*n+j]*scv_5+scv_8;
    }
    scv_9=a*scv_7+b*scv_9;
    scv_6=a*scv_10+b*scv_6;
    scv_8=a*scv_11+b*scv_8;
    y[(i+1)]=scv_6;
    y[(i+2)]=scv_8;
    y[i]=scv_9;
  }
  for (i=n-((n-1)%3)-1; i<=n-1; i=i+1) {
    double scv_17, scv_18;
    scv_17=y[i];
    scv_18=0;
    scv_17=0;
    {
      for (j=0; j<=n-4; j=j+4) {
        double scv_12, scv_13, scv_14, scv_15;
        scv_12=x[j];
        scv_13=x[(j+3)];
        scv_14=x[(j+2)];
        scv_15=x[(j+1)];
        scv_18=A[i*n+j]*scv_12+scv_18;
        scv_18=A[i*n+j+1]*scv_15+scv_18;
        scv_18=A[i*n+j+2]*scv_14+scv_18;
        scv_18=A[i*n+j+3]*scv_13+scv_18;
        scv_17=B[i*n+j]*scv_12+scv_17;
        scv_17=B[i*n+j+1]*scv_15+scv_17;
        scv_17=B[i*n+j+2]*scv_14+scv_17;
        scv_17=B[i*n+j+3]*scv_13+scv_17;
      }
      for (; j<=n-1; j=j+1) {
        double scv_16;
        scv_16=x[j];
        scv_18=A[i*n+j]*scv_16+scv_18;
        scv_17=B[i*n+j]*scv_16+scv_17;
      }
    }
    scv_17=a*scv_18+b*scv_17;
    y[i]=scv_17;
  }
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

