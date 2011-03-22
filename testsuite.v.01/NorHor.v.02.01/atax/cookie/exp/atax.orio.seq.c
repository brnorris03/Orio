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
  
{
  register int cbv_1;
  cbv_1=ny-1;
#pragma ivdep
#pragma vector always
  for (i=0; i<=cbv_1; i=i+1) 
    y[i]=0.0;
}
{
  for (i=0; i<=nx-5; i=i+5) {
    double scv_11, scv_12, scv_13, scv_14, scv_15;
    scv_15=0;
    scv_13=0;
    scv_11=0;
    scv_14=0;
    scv_12=0;
    register int cbv_1;
    cbv_1=ny-4;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_1; j=j+4) {
      double scv_1, scv_2, scv_3, scv_4;
      scv_1=x[j];
      scv_2=x[(j+3)];
      scv_3=x[(j+1)];
      scv_4=x[(j+2)];
      scv_15=scv_15+A[i*ny+j]*scv_1;
      scv_13=scv_13+A[(i+1)*ny+j]*scv_1;
      scv_11=scv_11+A[(i+2)*ny+j]*scv_1;
      scv_14=scv_14+A[(i+3)*ny+j]*scv_1;
      scv_12=scv_12+A[(i+4)*ny+j]*scv_1;
      scv_15=scv_15+A[i*ny+j+1]*scv_3;
      scv_13=scv_13+A[(i+1)*ny+j+1]*scv_3;
      scv_11=scv_11+A[(i+2)*ny+j+1]*scv_3;
      scv_14=scv_14+A[(i+3)*ny+j+1]*scv_3;
      scv_12=scv_12+A[(i+4)*ny+j+1]*scv_3;
      scv_15=scv_15+A[i*ny+j+2]*scv_4;
      scv_13=scv_13+A[(i+1)*ny+j+2]*scv_4;
      scv_11=scv_11+A[(i+2)*ny+j+2]*scv_4;
      scv_14=scv_14+A[(i+3)*ny+j+2]*scv_4;
      scv_12=scv_12+A[(i+4)*ny+j+2]*scv_4;
      scv_15=scv_15+A[i*ny+j+3]*scv_2;
      scv_13=scv_13+A[(i+1)*ny+j+3]*scv_2;
      scv_11=scv_11+A[(i+2)*ny+j+3]*scv_2;
      scv_14=scv_14+A[(i+3)*ny+j+3]*scv_2;
      scv_12=scv_12+A[(i+4)*ny+j+3]*scv_2;
    }
    register int cbv_2;
    cbv_2=ny-1;
#pragma ivdep
#pragma vector always
    for (; j<=cbv_2; j=j+1) {
      double scv_5;
      scv_5=x[j];
      scv_15=scv_15+A[i*ny+j]*scv_5;
      scv_13=scv_13+A[(i+1)*ny+j]*scv_5;
      scv_11=scv_11+A[(i+2)*ny+j]*scv_5;
      scv_14=scv_14+A[(i+3)*ny+j]*scv_5;
      scv_12=scv_12+A[(i+4)*ny+j]*scv_5;
    }
    register int cbv_3;
    cbv_3=ny-4;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_3; j=j+4) {
      double scv_6, scv_7, scv_8, scv_9;
      scv_6=y[(j+3)];
      scv_7=y[j];
      scv_8=y[(j+1)];
      scv_9=y[(j+2)];
      scv_7=scv_7+A[i*ny+j]*scv_15;
      scv_7=scv_7+A[(i+1)*ny+j]*scv_13;
      scv_7=scv_7+A[(i+2)*ny+j]*scv_11;
      scv_7=scv_7+A[(i+3)*ny+j]*scv_14;
      scv_7=scv_7+A[(i+4)*ny+j]*scv_12;
      scv_8=scv_8+A[i*ny+j+1]*scv_15;
      scv_8=scv_8+A[(i+1)*ny+j+1]*scv_13;
      scv_8=scv_8+A[(i+2)*ny+j+1]*scv_11;
      scv_8=scv_8+A[(i+3)*ny+j+1]*scv_14;
      scv_8=scv_8+A[(i+4)*ny+j+1]*scv_12;
      scv_9=scv_9+A[i*ny+j+2]*scv_15;
      scv_9=scv_9+A[(i+1)*ny+j+2]*scv_13;
      scv_9=scv_9+A[(i+2)*ny+j+2]*scv_11;
      scv_9=scv_9+A[(i+3)*ny+j+2]*scv_14;
      scv_9=scv_9+A[(i+4)*ny+j+2]*scv_12;
      scv_6=scv_6+A[i*ny+j+3]*scv_15;
      scv_6=scv_6+A[(i+1)*ny+j+3]*scv_13;
      scv_6=scv_6+A[(i+2)*ny+j+3]*scv_11;
      scv_6=scv_6+A[(i+3)*ny+j+3]*scv_14;
      scv_6=scv_6+A[(i+4)*ny+j+3]*scv_12;
      y[(j+3)]=scv_6;
      y[j]=scv_7;
      y[(j+1)]=scv_8;
      y[(j+2)]=scv_9;
    }
    register int cbv_4;
    cbv_4=ny-1;
#pragma ivdep
#pragma vector always
    for (; j<=cbv_4; j=j+1) {
      double scv_10;
      scv_10=y[j];
      scv_10=scv_10+A[i*ny+j]*scv_15;
      scv_10=scv_10+A[(i+1)*ny+j]*scv_13;
      scv_10=scv_10+A[(i+2)*ny+j]*scv_11;
      scv_10=scv_10+A[(i+3)*ny+j]*scv_14;
      scv_10=scv_10+A[(i+4)*ny+j]*scv_12;
      y[j]=scv_10;
    }
  }
  for (; i<=nx-1; i=i+1) {
    double scv_21;
    scv_21=0;
    {
      register int cbv_5;
      cbv_5=ny-4;
#pragma ivdep
#pragma vector always
      for (j=0; j<=cbv_5; j=j+4) {
        scv_21=scv_21+A[i*ny+j]*x[j];
        scv_21=scv_21+A[i*ny+j+1]*x[(j+1)];
        scv_21=scv_21+A[i*ny+j+2]*x[(j+2)];
        scv_21=scv_21+A[i*ny+j+3]*x[(j+3)];
      }
      register int cbv_6;
      cbv_6=ny-1;
#pragma ivdep
#pragma vector always
      for (; j<=cbv_6; j=j+1) {
        scv_21=scv_21+A[i*ny+j]*x[j];
      }
    }
    {
      register int cbv_7;
      cbv_7=ny-4;
#pragma ivdep
#pragma vector always
      for (j=0; j<=cbv_7; j=j+4) {
        double scv_16, scv_17, scv_18, scv_19;
        scv_16=y[(j+3)];
        scv_17=y[(j+1)];
        scv_18=y[(j+2)];
        scv_19=y[j];
        scv_19=scv_19+A[i*ny+j]*scv_21;
        scv_17=scv_17+A[i*ny+j+1]*scv_21;
        scv_18=scv_18+A[i*ny+j+2]*scv_21;
        scv_16=scv_16+A[i*ny+j+3]*scv_21;
        y[(j+3)]=scv_16;
        y[(j+1)]=scv_17;
        y[(j+2)]=scv_18;
        y[j]=scv_19;
      }
      register int cbv_8;
      cbv_8=ny-1;
#pragma ivdep
#pragma vector always
      for (; j<=cbv_8; j=j+1) {
        double scv_20;
        scv_20=y[j];
        scv_20=scv_20+A[i*ny+j]*scv_21;
        y[j]=scv_20;
      }
    }
  }
}



  



#else

int i,j;
  
{
  register int cbv_1;
  cbv_1=ny-6;
#pragma ivdep
#pragma vector always
  for (i=0; i<=cbv_1; i=i+6) {
    y[i]=0.0;
    y[(i+1)]=0.0;
    y[(i+2)]=0.0;
    y[(i+3)]=0.0;
    y[(i+4)]=0.0;
    y[(i+5)]=0.0;
  }
  register int cbv_2;
  cbv_2=ny-1;
#pragma ivdep
#pragma vector always
  for (; i<=cbv_2; i=i+1) 
    y[i]=0.0;
}
{
  for (i=0; i<=nx-8; i=i+8) {
    double scv_3, scv_4, scv_5, scv_6, scv_7, scv_8, scv_9, scv_10;
    scv_5=0;
    scv_3=0;
    scv_9=0;
    scv_4=0;
    scv_10=0;
    scv_8=0;
    scv_7=0;
    scv_6=0;
    register int cbv_1;
    cbv_1=ny-1;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_1; j=j+1) {
      double scv_1;
      scv_1=x[j];
      scv_5=scv_5+A[i][j]*scv_1;
      scv_3=scv_3+A[(i+1)][j]*scv_1;
      scv_9=scv_9+A[(i+2)][j]*scv_1;
      scv_4=scv_4+A[(i+3)][j]*scv_1;
      scv_10=scv_10+A[(i+4)][j]*scv_1;
      scv_8=scv_8+A[(i+5)][j]*scv_1;
      scv_7=scv_7+A[(i+6)][j]*scv_1;
      scv_6=scv_6+A[(i+7)][j]*scv_1;
    }
    register int cbv_2;
    cbv_2=ny-1;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_2; j=j+1) {
      double scv_2;
      scv_2=y[j];
      scv_2=scv_2+A[i][j]*scv_5;
      scv_2=scv_2+A[(i+1)][j]*scv_3;
      scv_2=scv_2+A[(i+2)][j]*scv_9;
      scv_2=scv_2+A[(i+3)][j]*scv_4;
      scv_2=scv_2+A[(i+4)][j]*scv_10;
      scv_2=scv_2+A[(i+5)][j]*scv_8;
      scv_2=scv_2+A[(i+6)][j]*scv_7;
      scv_2=scv_2+A[(i+7)][j]*scv_6;
      y[j]=scv_2;
    }
  }
  for (; i<=nx-1; i=i+1) {
    double scv_12;
    scv_12=0;
    register int cbv_3;
    cbv_3=ny-1;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_3; j=j+1) {
      scv_12=scv_12+A[i][j]*x[j];
    }
    register int cbv_4;
    cbv_4=ny-1;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_4; j=j+1) {
      double scv_11;
      scv_11=y[j];
      scv_11=scv_11+A[i][j]*scv_12;
      y[j]=scv_11;
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

