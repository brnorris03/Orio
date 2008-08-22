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
  cbv_1=ny-4;
#pragma ivdep
#pragma vector always
  for (i=0; i<=cbv_1; i=i+4) {
    y[i]=0.0;
    y[(i+1)]=0.0;
    y[(i+2)]=0.0;
    y[(i+3)]=0.0;
  }
  register int cbv_2;
  cbv_2=ny-1;
#pragma ivdep
#pragma vector always
  for (; i<=cbv_2; i=i+1) 
    y[i]=0.0;
}
{
#pragma omp parallel for private(j,i)
  for (i=0; i<=nx-24; i=i+24) {
    double scv_3, scv_4, scv_5, scv_6, scv_7, scv_8, scv_9, scv_10;
    double scv_11, scv_12, scv_13, scv_14, scv_15, scv_16, scv_17, scv_18;
    double scv_19, scv_20, scv_21, scv_22, scv_23, scv_24, scv_25, scv_26;
    scv_10=0;
    scv_23=0;
    scv_14=0;
    scv_9=0;
    scv_20=0;
    scv_13=0;
    scv_12=0;
    scv_11=0;
    scv_22=0;
    scv_21=0;
    scv_15=0;
    scv_19=0;
    scv_4=0;
    scv_5=0;
    scv_16=0;
    scv_17=0;
    scv_18=0;
    scv_26=0;
    scv_8=0;
    scv_7=0;
    scv_3=0;
    scv_6=0;
    scv_25=0;
    scv_24=0;
    register int cbv_1;
    cbv_1=ny-1;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_1; j=j+1) {
      double scv_1;
      scv_1=x[j];
      scv_10=scv_10+A[i*ny+j]*scv_1;
      scv_23=scv_23+A[(i+1)*ny+j]*scv_1;
      scv_14=scv_14+A[(i+2)*ny+j]*scv_1;
      scv_9=scv_9+A[(i+3)*ny+j]*scv_1;
      scv_20=scv_20+A[(i+4)*ny+j]*scv_1;
      scv_13=scv_13+A[(i+5)*ny+j]*scv_1;
      scv_12=scv_12+A[(i+6)*ny+j]*scv_1;
      scv_11=scv_11+A[(i+7)*ny+j]*scv_1;
      scv_22=scv_22+A[(i+8)*ny+j]*scv_1;
      scv_21=scv_21+A[(i+9)*ny+j]*scv_1;
      scv_15=scv_15+A[(i+10)*ny+j]*scv_1;
      scv_19=scv_19+A[(i+11)*ny+j]*scv_1;
      scv_4=scv_4+A[(i+12)*ny+j]*scv_1;
      scv_5=scv_5+A[(i+13)*ny+j]*scv_1;
      scv_16=scv_16+A[(i+14)*ny+j]*scv_1;
      scv_17=scv_17+A[(i+15)*ny+j]*scv_1;
      scv_18=scv_18+A[(i+16)*ny+j]*scv_1;
      scv_26=scv_26+A[(i+17)*ny+j]*scv_1;
      scv_8=scv_8+A[(i+18)*ny+j]*scv_1;
      scv_7=scv_7+A[(i+19)*ny+j]*scv_1;
      scv_3=scv_3+A[(i+20)*ny+j]*scv_1;
      scv_6=scv_6+A[(i+21)*ny+j]*scv_1;
      scv_25=scv_25+A[(i+22)*ny+j]*scv_1;
      scv_24=scv_24+A[(i+23)*ny+j]*scv_1;
    }
    register int cbv_2;
    cbv_2=ny-1;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_2; j=j+1) {
      double scv_2;
      scv_2=y[j];
      scv_2=scv_2+A[i*ny+j]*scv_10;
      scv_2=scv_2+A[(i+1)*ny+j]*scv_23;
      scv_2=scv_2+A[(i+2)*ny+j]*scv_14;
      scv_2=scv_2+A[(i+3)*ny+j]*scv_9;
      scv_2=scv_2+A[(i+4)*ny+j]*scv_20;
      scv_2=scv_2+A[(i+5)*ny+j]*scv_13;
      scv_2=scv_2+A[(i+6)*ny+j]*scv_12;
      scv_2=scv_2+A[(i+7)*ny+j]*scv_11;
      scv_2=scv_2+A[(i+8)*ny+j]*scv_22;
      scv_2=scv_2+A[(i+9)*ny+j]*scv_21;
      scv_2=scv_2+A[(i+10)*ny+j]*scv_15;
      scv_2=scv_2+A[(i+11)*ny+j]*scv_19;
      scv_2=scv_2+A[(i+12)*ny+j]*scv_4;
      scv_2=scv_2+A[(i+13)*ny+j]*scv_5;
      scv_2=scv_2+A[(i+14)*ny+j]*scv_16;
      scv_2=scv_2+A[(i+15)*ny+j]*scv_17;
      scv_2=scv_2+A[(i+16)*ny+j]*scv_18;
      scv_2=scv_2+A[(i+17)*ny+j]*scv_26;
      scv_2=scv_2+A[(i+18)*ny+j]*scv_8;
      scv_2=scv_2+A[(i+19)*ny+j]*scv_7;
      scv_2=scv_2+A[(i+20)*ny+j]*scv_3;
      scv_2=scv_2+A[(i+21)*ny+j]*scv_6;
      scv_2=scv_2+A[(i+22)*ny+j]*scv_25;
      scv_2=scv_2+A[(i+23)*ny+j]*scv_24;
      y[j]=scv_2;
    }

  }
  for (i=nx-((nx-1)%24)-1; i<=nx-1; i=i+1) {
    double scv_28;
    scv_28=0;
    register int cbv_3;
    cbv_3=ny-1;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_3; j=j+1) {
      scv_28=scv_28+A[i*ny+j]*x[j];
    }
    register int cbv_4;
    cbv_4=ny-1;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_4; j=j+1) {
      double scv_27;
      scv_27=y[j];
      scv_27=scv_27+A[i*ny+j]*scv_28;
      y[j]=scv_27;
    }
  }
}



  


#else

int i,j;
  
{
  register int cbv_1;
  cbv_1=ny-5;
#pragma ivdep
#pragma vector always
  for (i=0; i<=cbv_1; i=i+5) {
    y[i]=0.0;
    y[(i+1)]=0.0;
    y[(i+2)]=0.0;
    y[(i+3)]=0.0;
    y[(i+4)]=0.0;
  }
  register int cbv_2;
  cbv_2=ny-1;
#pragma ivdep
#pragma vector always
  for (; i<=cbv_2; i=i+1) 
    y[i]=0.0;
}
{
#pragma omp parallel for private(j,i)
  for (i=0; i<=nx-24; i=i+24) {
    double scv_3, scv_4, scv_5, scv_6, scv_7, scv_8, scv_9, scv_10;
    double scv_11, scv_12, scv_13, scv_14, scv_15, scv_16, scv_17, scv_18;
    double scv_19, scv_20, scv_21, scv_22, scv_23, scv_24, scv_25, scv_26;
    scv_10=0;
    scv_23=0;
    scv_14=0;
    scv_9=0;
    scv_20=0;
    scv_13=0;
    scv_12=0;
    scv_11=0;
    scv_22=0;
    scv_21=0;
    scv_15=0;
    scv_19=0;
    scv_4=0;
    scv_5=0;
    scv_16=0;
    scv_17=0;
    scv_18=0;
    scv_26=0;
    scv_8=0;
    scv_7=0;
    scv_3=0;
    scv_6=0;
    scv_25=0;
    scv_24=0;
    for (j=0; j<=ny-1; j=j+1) {
      double scv_1;
      scv_1=x[j];
      scv_10=scv_10+A[i][j]*scv_1;
      scv_23=scv_23+A[(i+1)][j]*scv_1;
      scv_14=scv_14+A[(i+2)][j]*scv_1;
      scv_9=scv_9+A[(i+3)][j]*scv_1;
      scv_20=scv_20+A[(i+4)][j]*scv_1;
      scv_13=scv_13+A[(i+5)][j]*scv_1;
      scv_12=scv_12+A[(i+6)][j]*scv_1;
      scv_11=scv_11+A[(i+7)][j]*scv_1;
      scv_22=scv_22+A[(i+8)][j]*scv_1;
      scv_21=scv_21+A[(i+9)][j]*scv_1;
      scv_15=scv_15+A[(i+10)][j]*scv_1;
      scv_19=scv_19+A[(i+11)][j]*scv_1;
      scv_4=scv_4+A[(i+12)][j]*scv_1;
      scv_5=scv_5+A[(i+13)][j]*scv_1;
      scv_16=scv_16+A[(i+14)][j]*scv_1;
      scv_17=scv_17+A[(i+15)][j]*scv_1;
      scv_18=scv_18+A[(i+16)][j]*scv_1;
      scv_26=scv_26+A[(i+17)][j]*scv_1;
      scv_8=scv_8+A[(i+18)][j]*scv_1;
      scv_7=scv_7+A[(i+19)][j]*scv_1;
      scv_3=scv_3+A[(i+20)][j]*scv_1;
      scv_6=scv_6+A[(i+21)][j]*scv_1;
      scv_25=scv_25+A[(i+22)][j]*scv_1;
      scv_24=scv_24+A[(i+23)][j]*scv_1;
    }
    for (j=0; j<=ny-1; j=j+1) {
      double scv_2;
      scv_2=y[j];
      scv_2=scv_2+A[i][j]*scv_10;
      scv_2=scv_2+A[(i+1)][j]*scv_23;
      scv_2=scv_2+A[(i+2)][j]*scv_14;
      scv_2=scv_2+A[(i+3)][j]*scv_9;
      scv_2=scv_2+A[(i+4)][j]*scv_20;
      scv_2=scv_2+A[(i+5)][j]*scv_13;
      scv_2=scv_2+A[(i+6)][j]*scv_12;
      scv_2=scv_2+A[(i+7)][j]*scv_11;
      scv_2=scv_2+A[(i+8)][j]*scv_22;
      scv_2=scv_2+A[(i+9)][j]*scv_21;
      scv_2=scv_2+A[(i+10)][j]*scv_15;
      scv_2=scv_2+A[(i+11)][j]*scv_19;
      scv_2=scv_2+A[(i+12)][j]*scv_4;
      scv_2=scv_2+A[(i+13)][j]*scv_5;
      scv_2=scv_2+A[(i+14)][j]*scv_16;
      scv_2=scv_2+A[(i+15)][j]*scv_17;
      scv_2=scv_2+A[(i+16)][j]*scv_18;
      scv_2=scv_2+A[(i+17)][j]*scv_26;
      scv_2=scv_2+A[(i+18)][j]*scv_8;
      scv_2=scv_2+A[(i+19)][j]*scv_7;
      scv_2=scv_2+A[(i+20)][j]*scv_3;
      scv_2=scv_2+A[(i+21)][j]*scv_6;
      scv_2=scv_2+A[(i+22)][j]*scv_25;
      scv_2=scv_2+A[(i+23)][j]*scv_24;
      y[j]=scv_2;
    }
  }
  for (i=nx-((nx-1)%24)-1; i<=nx-1; i=i+1) {
    double scv_28;
    scv_28=0;
    for (j=0; j<=ny-1; j=j+1) {
      scv_28=scv_28+A[i][j]*x[j];
    }
    for (j=0; j<=ny-1; j=j+1) {
      double scv_27;
      scv_27=y[j];
      scv_27=scv_27+A[i][j]*scv_28;
      y[j]=scv_27;
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
