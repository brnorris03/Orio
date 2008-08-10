#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


double *x1;
double *x2;
double *x3;
double *x4;
double *x5;
double *y;
double a1;
double a2;
double a3;
double a4;
double a5;

void malloc_arrays() {
    int i1;
    x1 = (double*) malloc((N) * sizeof(double));
    x2 = (double*) malloc((N) * sizeof(double));
    x3 = (double*) malloc((N) * sizeof(double));
    x4 = (double*) malloc((N) * sizeof(double));
    x5 = (double*) malloc((N) * sizeof(double));
    y = (double*) malloc((N) * sizeof(double));
}

void init_input_vars() {
    int i1;
    for (i1=0; i1<N; i1++) {
        x1[i1] = (i1+1) % 4 + 1;
        x2[i1] = (i1+5) % 10 + 1;
        x3[i1] = (i1+3) % 6 + 1;
        x4[i1] = (i1+9) % 9 + 1;
        x5[i1] = (i1+8) % 15 + 1;
        y[i1] = 0;
    }
    a1 = (double) 6;
    a2 = (double) 7;
    a3 = (double) 4;
    a4 = (double) 1;
    a5 = (double) 9;
}

double getClock()
{
    struct timezone tzp;
    struct timeval tp;
    gettimeofday (&tp, &tzp);
    return (tp.tv_sec + tp.tv_usec*1.0e-6);
}

int main(int argc, char *argv[])
{
    malloc_arrays();
    init_input_vars();

    int one = 1;
    int n = N;

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
	for (i=0; i<=n-1; i++)
            y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];

    }
    orio_t_end = getClock();
    orio_t_total = orio_t_end - orio_t_start;

    orio_t_total = orio_t_total / REPS; 
    double mflops = (10.0*N)/(orio_t_total*1000000);

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
    printf("%.6f\t%.3f\n", orio_t_total, mflops);
#endif

    return y[0];
}

