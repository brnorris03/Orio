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
      for (i=0; i<=ny-1; i=i+1) 
	s[i]=0;
      {
	for (i=0; i<=nx-4; i=i+4) {
	  double scv_19, scv_20, scv_21, scv_22, scv_23, scv_24, scv_25, scv_26;
	  scv_19=r[(i+2)];
	  scv_20=r[(i+3)];
	  scv_24=r[i];
	  scv_25=r[(i+1)];
	  scv_23=0;
	  scv_26=0;
	  scv_22=0;
	  scv_21=0;
	  double* tA0=A+(i)*ny;
	  double* tA1=A+(i+1)*ny;
	  double* tA2=A+(i+2)*ny;
	  double* tA3=A+(i+3)*ny;
	  for (j=0; j<=ny-2; j=j+2) {
	    double scv_1, scv_2, scv_3, scv_4, scv_5, scv_6, scv_7, scv_8;
	    double scv_9, scv_10, scv_11, scv_12;
	    scv_1=p[(j+1)];
	    scv_2=tA2[j+1];
	    scv_3=tA2[j];
	    scv_4=tA0[j+1];
	    scv_5=tA1[j+1];
	    scv_6=p[j];
	    scv_7=tA3[j+1];
	    scv_8=s[(j+1)];
	    scv_9=tA0[j];
	    scv_10=tA1[j];
	    scv_11=s[j];
	    scv_12=tA3[j];
	    scv_11=scv_11+scv_24*scv_9;
	    scv_11=scv_11+scv_25*scv_10;
	    scv_11=scv_11+scv_19*scv_3;
	    scv_11=scv_11+scv_20*scv_12;
	    scv_8=scv_8+scv_24*scv_4;
	    scv_8=scv_8+scv_25*scv_5;
	    scv_8=scv_8+scv_19*scv_2;
	    scv_8=scv_8+scv_20*scv_7;
	    scv_23=scv_23+scv_9*scv_6;
	    scv_26=scv_26+scv_10*scv_6;
	    scv_22=scv_22+scv_3*scv_6;
	    scv_21=scv_21+scv_12*scv_6;
	    scv_23=scv_23+scv_4*scv_1;
	    scv_26=scv_26+scv_5*scv_1;
	    scv_22=scv_22+scv_2*scv_1;
	    scv_21=scv_21+scv_7*scv_1;
	    s[(j+1)]=scv_8;
	    s[j]=scv_11;
	  }
	  for (; j<=ny-1; j=j+1) {
	    double scv_13, scv_14, scv_15, scv_16, scv_17, scv_18;
	    scv_13=tA2[j];
	    scv_14=p[j];
	    scv_15=tA0[j];
	    scv_16=tA1[j];
	    scv_17=s[j];
	    scv_18=tA3[j];
	    scv_17=scv_17+scv_24*scv_15;
	    scv_17=scv_17+scv_25*scv_16;
	    scv_17=scv_17+scv_19*scv_13;
	    scv_17=scv_17+scv_20*scv_18;
	    scv_23=scv_23+scv_15*scv_14;
	    scv_26=scv_26+scv_16*scv_14;
	    scv_22=scv_22+scv_13*scv_14;
	    scv_21=scv_21+scv_18*scv_14;
	    s[j]=scv_17;
	  }
	  q[(i+3)]=scv_21;
	  q[(i+2)]=scv_22;
	  q[i]=scv_23;
	  q[(i+1)]=scv_26;
	}
	for (; i<=nx-1; i=i+1) {
	  double scv_33, scv_34;
	  scv_33=q[i];
	  scv_34=r[i];
	  scv_33=0;
	  double* tA=A+(i)*ny;	  
	  {
	    for (j=0; j<=ny-2; j=j+2) {
	      double scv_27, scv_28, scv_29, scv_30;
	      scv_27=tA[j+1];
	      scv_28=tA[j];
	      scv_29=s[j];
	      scv_30=s[(j+1)];
	      scv_29=scv_29+scv_34*scv_28;
	      scv_30=scv_30+scv_34*scv_27;
	      scv_33=scv_33+scv_28*p[j];
	      scv_33=scv_33+scv_27*p[(j+1)];
	      s[j]=scv_29;
	      s[(j+1)]=scv_30;
	    }
	    for (; j<=ny-1; j=j+1) {
	      double scv_31, scv_32;
	      scv_31=s[j];
	      scv_32=tA[j];
	      scv_31=scv_31+scv_34*scv_32;
	      scv_33=scv_33+scv_32*p[j];
	      s[j]=scv_31;
	    }
	  }
	  q[i]=scv_33;
	}
      }
#else
      int i,j;
      {
	register int cbv_1;
	cbv_1=ny-1;
#pragma ivdep
#pragma vector always
	for (i=0; i<=cbv_1; i=i+1) 
	  s[i]=0;
      }
      {
	for (i=0; i<=nx-12; i=i+12) {
	  q[i]=0;
	  q[(i+1)]=0;
	  q[(i+2)]=0;
	  q[(i+3)]=0;
	  q[(i+4)]=0;
	  q[(i+5)]=0;
	  q[(i+6)]=0;
	  q[(i+7)]=0;
	  q[(i+8)]=0;
	  q[(i+9)]=0;
	  q[(i+10)]=0;
	  q[(i+11)]=0;
	  register int cbv_1;
	  cbv_1=ny-1;
#pragma ivdep
#pragma vector always
	  for (j=0; j<=cbv_1; j=j+1) {
	    s[j]=s[j]+r[i]*A[i][j];
	    s[j]=s[j]+r[(i+1)]*A[(i+1)][j];
	    s[j]=s[j]+r[(i+2)]*A[(i+2)][j];
	    s[j]=s[j]+r[(i+3)]*A[(i+3)][j];
	    s[j]=s[j]+r[(i+4)]*A[(i+4)][j];
	    s[j]=s[j]+r[(i+5)]*A[(i+5)][j];
	    s[j]=s[j]+r[(i+6)]*A[(i+6)][j];
	    s[j]=s[j]+r[(i+7)]*A[(i+7)][j];
	    s[j]=s[j]+r[(i+8)]*A[(i+8)][j];
	    s[j]=s[j]+r[(i+9)]*A[(i+9)][j];
	    s[j]=s[j]+r[(i+10)]*A[(i+10)][j];
	    s[j]=s[j]+r[(i+11)]*A[(i+11)][j];
	    q[i]=q[i]+A[i][j]*p[j];
	    q[(i+1)]=q[(i+1)]+A[(i+1)][j]*p[j];
	    q[(i+2)]=q[(i+2)]+A[(i+2)][j]*p[j];
	    q[(i+3)]=q[(i+3)]+A[(i+3)][j]*p[j];
	    q[(i+4)]=q[(i+4)]+A[(i+4)][j]*p[j];
	    q[(i+5)]=q[(i+5)]+A[(i+5)][j]*p[j];
	    q[(i+6)]=q[(i+6)]+A[(i+6)][j]*p[j];
	    q[(i+7)]=q[(i+7)]+A[(i+7)][j]*p[j];
	    q[(i+8)]=q[(i+8)]+A[(i+8)][j]*p[j];
	    q[(i+9)]=q[(i+9)]+A[(i+9)][j]*p[j];
	    q[(i+10)]=q[(i+10)]+A[(i+10)][j]*p[j];
	    q[(i+11)]=q[(i+11)]+A[(i+11)][j]*p[j];
	  }
	}
	for (; i<=nx-1; i=i+1) {
	  q[i]=0;
	  register int cbv_2;
	  cbv_2=ny-1;
#pragma ivdep
#pragma vector always
	  for (j=0; j<=cbv_2; j=j+1) {
	    s[j]=s[j]+r[i]*A[i][j];
	    q[i]=q[i]+A[i][j]*p[j];
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

