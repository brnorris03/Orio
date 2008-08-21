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
	  for (i=0; i<=n-5; i=i+5) {
	    double* tB0=B+i*n;
	    double* tB1=B+(i+1)*n;
	    double* tB2=B+(i+2)*n;
	    double* tB3=B+(i+3)*n;
	    double* tB4=B+(i+4)*n;
	    for (j=0; j<=n-1; j=j+1) {
	      w[i]=w[i]+tB0[j]*x[j];
	      w[(i+1)]=w[(i+1)]+tB1[j]*x[j];
	      w[(i+2)]=w[(i+2)]+tB2[j]*x[j];
	      w[(i+3)]=w[(i+3)]+tB3[j]*x[j];
	      w[(i+4)]=w[(i+4)]+tB4[j]*x[j];
	    }
	    w[i]=a*w[i];
	    w[(i+1)]=a*w[(i+1)];
	    w[(i+2)]=a*w[(i+2)];
	    w[(i+3)]=a*w[(i+3)];
	    w[(i+4)]=a*w[(i+4)];
	  }
	  for (; i<=n-1; i=i+1) {
	    double* tB=B+i*n;
	    for (j=0; j<=n-1; j=j+1) {
	      w[i]=w[i]+tB[j]*x[j];
	    }
	    w[i]=a*w[i];
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
