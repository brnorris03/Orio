#include <stdio.h>
#include <stdlib.h>
#include "consts.h"
#include <sys/time.h>



double getclock(){
  struct timezone tzp;
  struct timeval tp;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}


void axpy1(int n, double *y, double a1, double *x1)
{
register int i;



   {
    for (i=0; i<=n-1; i++) {
    	y[i]=y[i]+a1*x1[i];
    }
    
   }
}

int main(){
	double* y = (double*) malloc(sizeof(double)*NN);
	double* x1 = (double*) malloc(sizeof(double)*NN);
	double a1 = AA;
	int i;
	for(i=0; i<NN; i++){
		y[i] = i;
		x1[i] = i;
	}
        double start = getclock();
	axpy1(NN, y, a1, x1);
        double end  = getclock();
        double passedTime = 1000*(end-start);
        printf("elapsed: %e ms\n",passedTime);
	for(i=0; i<13; i++)
		printf("%f\n", y[i]);
        for(i=NN-9; i<NN; i++)
                printf("%f\n", y[i]);

	return 0;
}
