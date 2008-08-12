#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

double getClock()
{
    struct timezone tzp;
    struct timeval tp;
    gettimeofday (&tp, &tzp);
    return (tp.tv_sec + tp.tv_usec*1.0e-6);
}

int main(int argc, char *argv[])
{
    double *y;
    double *x1;
    double *x2;
    double *x3;
    double *x4;
    double a1;
    double a2;
    double a3;
    double a4;

    int n = N;
    {
	int i1;
	y = (double*) malloc((n) * sizeof(double));
	x1 = (double*) malloc((n) * sizeof(double));
	x2 = (double*) malloc((n) * sizeof(double));
	x3 = (double*) malloc((n) * sizeof(double));
	x4 = (double*) malloc((n) * sizeof(double));
	for (i1=0; i1<n; i1++) {
	    x1[i1] = (i1+1) % 4 + 1;
	    x2[i1] = (i1+5) % 10 + 1;
	    x3[i1] = (i1+3) % 6 + 1;
	    x4[i1] = (i1+9) % 9 + 1;
	    y[i1] = 0;
	}
	a1 = (double) 6;
	a2 = (double) 7;
	a3 = (double) 4;
	a4 = (double) 1;
    }

    double orio_t_start, orio_t_end, orio_t_total=0;
    int orio_i;
    int reps = REPS;
#ifdef TEST
    reps = 1;
#endif

    orio_t_start = getClock(); 
    for (orio_i=0; orio_i<reps; orio_i++)
    {
	int i;
#pragma omp parallel for
	for (i=0; i<=n-1; i++)
	    y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i];

    }
    orio_t_end = getClock();
    orio_t_total = orio_t_end - orio_t_start;

    orio_t_total = orio_t_total / REPS; 
    double mflops = (8.0*N)/(orio_t_total*1000000);

#ifdef TEST
    {
	int i;
	for (i=0; i<=n-1; i++) {
	    if (i%10 == 0)
		printf("\n");
	    printf("%f ",y[i]);
	}
    }
#else
    printf("%f\t%f\n", orio_t_total, mflops);
#endif

    return y[0];
}

