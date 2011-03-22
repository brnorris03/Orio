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
	for (i=0; i<=n-1; i=i+1) {
	  x[i]=0;
	  w[i]=0;
	}
	{
	  for (j=0; j<=n-4; j=j+4) {
	    for (i=0; i<=n-1; i=i+1) {
	      double scv_1, scv_2, scv_3, scv_4, scv_5, scv_6, scv_7;
	      scv_1=B[j*n+i];
	      scv_2=v2[i];
	      scv_3=B[(j+2)*n+i];
	      scv_4=B[(j+1)*n+i];
	      scv_5=x[i];
	      scv_6=B[(j+3)*n+i];
	      scv_7=v1[i];
	      scv_1=u2[j]*scv_2+u1[j]*scv_7+A[j*n+i];
	      scv_4=u2[(j+1)]*scv_2+u1[(j+1)]*scv_7+A[(j+1)*n+i];
	      scv_3=u2[(j+2)]*scv_2+u1[(j+2)]*scv_7+A[(j+2)*n+i];
	      scv_6=u2[(j+3)]*scv_2+u1[(j+3)]*scv_7+A[(j+3)*n+i];
	      scv_5=y[j]*scv_1+scv_5;
	      scv_5=y[(j+1)]*scv_4+scv_5;
	      scv_5=y[(j+2)]*scv_3+scv_5;
	      scv_5=y[(j+3)]*scv_6+scv_5;
	      B[j*n+i]=scv_1;
	      B[(j+2)*n+i]=scv_3;
	      B[(j+1)*n+i]=scv_4;
	      x[i]=scv_5;
	      B[(j+3)*n+i]=scv_6;
	    }
	  }
	  for (; j<=n-1; j=j+1) {
	    for (i=0; i<=n-1; i=i+1) {
	      double scv_8, scv_9;
	      scv_8=x[i];
	      scv_9=B[j*n+i];
	      scv_9=u2[j]*v2[i]+u1[j]*v1[i]+A[j*n+i];
	      scv_8=y[j]*scv_9+scv_8;
	      x[i]=scv_8;
	      B[j*n+i]=scv_9;
	    }
	  }
	}
	for (i=0; i<=n-1; i=i+1) {
	  x[i]=b*x[i]+z[i];
	}
	{
	  for (i=0; i<=n-4; i=i+4) {
	    double scv_6, scv_7, scv_8, scv_9;
	    scv_6=w[i];
	    scv_7=w[(i+3)];
	    scv_8=w[(i+1)];
	    scv_9=w[(i+2)];
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
	      scv_8=scv_8+B[(i+1)*n+j]*scv_1;
	      scv_9=scv_9+B[(i+2)*n+j]*scv_1;
	      scv_7=scv_7+B[(i+3)*n+j]*scv_1;
	      scv_6=scv_6+B[i*n+j+1]*scv_4;
	      scv_8=scv_8+B[(i+1)*n+j+1]*scv_4;
	      scv_9=scv_9+B[(i+2)*n+j+1]*scv_4;
	      scv_7=scv_7+B[(i+3)*n+j+1]*scv_4;
	      scv_6=scv_6+B[i*n+j+2]*scv_3;
	      scv_8=scv_8+B[(i+1)*n+j+2]*scv_3;
	      scv_9=scv_9+B[(i+2)*n+j+2]*scv_3;
	      scv_7=scv_7+B[(i+3)*n+j+2]*scv_3;
	      scv_6=scv_6+B[i*n+j+3]*scv_2;
	      scv_8=scv_8+B[(i+1)*n+j+3]*scv_2;
	      scv_9=scv_9+B[(i+2)*n+j+3]*scv_2;
	      scv_7=scv_7+B[(i+3)*n+j+3]*scv_2;
	    }
	    register int cbv_2;
	    cbv_2=n-1;
#pragma ivdep
#pragma vector always
	    for (; j<=cbv_2; j=j+1) {
	      double scv_5;
	      scv_5=x[j];
	      scv_6=scv_6+B[i*n+j]*scv_5;
	      scv_8=scv_8+B[(i+1)*n+j]*scv_5;
	      scv_9=scv_9+B[(i+2)*n+j]*scv_5;
	      scv_7=scv_7+B[(i+3)*n+j]*scv_5;
	    }
	    scv_6=a*scv_6;
	    scv_8=a*scv_8;
	    scv_9=a*scv_9;
	    scv_7=a*scv_7;
	    w[i]=scv_6;
	    w[(i+3)]=scv_7;
	    w[(i+1)]=scv_8;
	    w[(i+2)]=scv_9;
	  }
	  for (; i<=n-1; i=i+1) {
	    double scv_10;
	    scv_10=w[i];
	    {
	      register int cbv_3;
	      cbv_3=n-4;
#pragma ivdep
#pragma vector always
	      for (j=0; j<=cbv_3; j=j+4) {
		scv_10=scv_10+B[i*n+j]*x[j];
		scv_10=scv_10+B[i*n+j+1]*x[(j+1)];
		scv_10=scv_10+B[i*n+j+2]*x[(j+2)];
		scv_10=scv_10+B[i*n+j+3]*x[(j+3)];
	      }
	      register int cbv_4;
	      cbv_4=n-1;
#pragma ivdep
#pragma vector always
	      for (; j<=cbv_4; j=j+1) {
		scv_10=scv_10+B[i*n+j]*x[j];
	      }
	    }
	    scv_10=a*scv_10;
	    w[i]=scv_10;
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
	  for (i=0; i<=n-5; i=i+5) {
	    for (j=0; j<=n-1; j=j+1) {
	      w[i]=w[i]+B[i][j]*x[j];
	      w[(i+1)]=w[(i+1)]+B[(i+1)][j]*x[j];
	      w[(i+2)]=w[(i+2)]+B[(i+2)][j]*x[j];
	      w[(i+3)]=w[(i+3)]+B[(i+3)][j]*x[j];
	      w[(i+4)]=w[(i+4)]+B[(i+4)][j]*x[j];
	    }
	    w[i]=a*w[i];
	    w[(i+1)]=a*w[(i+1)];
	    w[(i+2)]=a*w[(i+2)];
	    w[(i+3)]=a*w[(i+3)];
	    w[(i+4)]=a*w[(i+4)];
	  }
	  for (; i<=n-1; i=i+1) {
	    for (j=0; j<=n-1; j=j+1) {
	      w[i]=w[i]+B[i][j]*x[j];
	    }
	    w[i]=a*w[i];
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
