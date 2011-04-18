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

for (i=0; i<=n-1; i=i+1) {
  x[i]=0;
  w[i]=0;
}
{
  for (j=0; j<=n-1; j=j+1) {
    double scv_9, scv_10, scv_11;
    scv_9=u2[j];
    scv_10=u1[j];
    scv_11=y[j];
    {
      for (i=0; i<=n-3; i=i+3) {
        double scv_1, scv_2, scv_3, scv_4, scv_5, scv_6;
        scv_1=x[(i+1)];
        scv_2=B[j*n+i];
        scv_3=x[(i+2)];
        scv_4=B[j*n+i+2];
        scv_5=B[j*n+i+1];
        scv_6=x[i];
        scv_2=scv_9*v2[i]+scv_10*v1[i]+A[j*n+i];
        scv_5=scv_9*v2[(i+1)]+scv_10*v1[(i+1)]+A[j*n+i+1];
        scv_4=scv_9*v2[(i+2)]+scv_10*v1[(i+2)]+A[j*n+i+2];
        scv_6=scv_11*scv_2+scv_6;
        scv_1=scv_11*scv_5+scv_1;
        scv_3=scv_11*scv_4+scv_3;
        x[(i+1)]=scv_1;
        B[j*n+i]=scv_2;
        x[(i+2)]=scv_3;
        B[j*n+i+2]=scv_4;
        B[j*n+i+1]=scv_5;
        x[i]=scv_6;
      }
      for (; i<=n-1; i=i+1) {
        double scv_7, scv_8;
        scv_7=x[i];
        scv_8=B[j*n+i];
        scv_8=scv_9*v2[i]+scv_10*v1[i]+A[j*n+i];
        scv_7=scv_11*scv_8+scv_7;
        x[i]=scv_7;
        B[j*n+i]=scv_8;
      }
    }
  }
}
{
#pragma omp parallel for private(i)
  for (i=0; i<=n-1; i=i+1) {
    x[i]=b*x[i]+z[i];
  }
}
{
#pragma omp parallel for private(j,i)
  for (i=0; i<=n-6; i=i+6) {
    double scv_6, scv_7, scv_8, scv_9, scv_10, scv_11;
    scv_6=w[i];
    scv_7=w[(i+1)];
    scv_8=w[(i+5)];
    scv_9=w[(i+4)];
    scv_10=w[(i+3)];
    scv_11=w[(i+2)];
    register int cbv_1;
    cbv_1=n-4;
#pragma ivdep
#pragma vector always
    for (j=0; j<=cbv_1; j=j+4) {
      double scv_1, scv_2, scv_3, scv_4;
      scv_1=x[j];
      scv_2=x[(j+3)];
      scv_3=x[(j+2)];
      scv_4=x[(j+1)];
      scv_6=scv_6+B[i*n+j]*scv_1;
      scv_7=scv_7+B[(i+1)*n+j]*scv_1;
      scv_11=scv_11+B[(i+2)*n+j]*scv_1;
      scv_10=scv_10+B[(i+3)*n+j]*scv_1;
      scv_9=scv_9+B[(i+4)*n+j]*scv_1;
      scv_8=scv_8+B[(i+5)*n+j]*scv_1;
      scv_6=scv_6+B[i*n+j+1]*scv_4;
      scv_7=scv_7+B[(i+1)*n+j+1]*scv_4;
      scv_11=scv_11+B[(i+2)*n+j+1]*scv_4;
      scv_10=scv_10+B[(i+3)*n+j+1]*scv_4;
      scv_9=scv_9+B[(i+4)*n+j+1]*scv_4;
      scv_8=scv_8+B[(i+5)*n+j+1]*scv_4;
      scv_6=scv_6+B[i*n+j+2]*scv_3;
      scv_7=scv_7+B[(i+1)*n+j+2]*scv_3;
      scv_11=scv_11+B[(i+2)*n+j+2]*scv_3;
      scv_10=scv_10+B[(i+3)*n+j+2]*scv_3;
      scv_9=scv_9+B[(i+4)*n+j+2]*scv_3;
      scv_8=scv_8+B[(i+5)*n+j+2]*scv_3;
      scv_6=scv_6+B[i*n+j+3]*scv_2;
      scv_7=scv_7+B[(i+1)*n+j+3]*scv_2;
      scv_11=scv_11+B[(i+2)*n+j+3]*scv_2;
      scv_10=scv_10+B[(i+3)*n+j+3]*scv_2;
      scv_9=scv_9+B[(i+4)*n+j+3]*scv_2;
      scv_8=scv_8+B[(i+5)*n+j+3]*scv_2;
    }
    register int cbv_2;
    cbv_2=n-1;
#pragma ivdep
#pragma vector always
    for (; j<=cbv_2; j=j+1) {
      double scv_5;
      scv_5=x[j];
      scv_6=scv_6+B[i*n+j]*scv_5;
      scv_7=scv_7+B[(i+1)*n+j]*scv_5;
      scv_11=scv_11+B[(i+2)*n+j]*scv_5;
      scv_10=scv_10+B[(i+3)*n+j]*scv_5;
      scv_9=scv_9+B[(i+4)*n+j]*scv_5;
      scv_8=scv_8+B[(i+5)*n+j]*scv_5;
    }
    scv_6=a*scv_6;
    scv_7=a*scv_7;
    scv_11=a*scv_11;
    scv_10=a*scv_10;
    scv_9=a*scv_9;
    scv_8=a*scv_8;
    w[i]=scv_6;
    w[(i+1)]=scv_7;
    w[(i+5)]=scv_8;
    w[(i+4)]=scv_9;
    w[(i+3)]=scv_10;
    w[(i+2)]=scv_11;
  }
  for (i=n-((n-1)%6)-1; i<=n-1; i=i+1) {
    double scv_12;
    scv_12=w[i];
    {
      register int cbv_3;
      cbv_3=n-4;
#pragma ivdep
#pragma vector always
      for (j=0; j<=cbv_3; j=j+4) {
        scv_12=scv_12+B[i*n+j]*x[j];
        scv_12=scv_12+B[i*n+j+1]*x[(j+1)];
        scv_12=scv_12+B[i*n+j+2]*x[(j+2)];
        scv_12=scv_12+B[i*n+j+3]*x[(j+3)];
      }
      register int cbv_4;
      cbv_4=n-1;
#pragma ivdep
#pragma vector always
      for (; j<=cbv_4; j=j+1) {
        scv_12=scv_12+B[i*n+j]*x[j];
      }
    }
    scv_12=a*scv_12;
    w[i]=scv_12;
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
#pragma omp parallel for private(i,j)
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
