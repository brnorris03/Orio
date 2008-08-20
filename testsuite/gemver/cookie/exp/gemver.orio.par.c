
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

double a;
double b;
int nx;
int ny;
double A[Nx][Ny];
double B[Nx][Ny];
double u1[Nx];
double u2[Nx];
double v1[Ny];
double v2[Ny];
double y[Nx];
double z[Ny];
double w[Nx];
double x[Ny];

void init_arrays()
{
  int i, j;
  a = 1.5;
  b = 2.5;
  nx = Nx;
  ny = Ny;
  for (i=0; i<=Nx-1; i++) {
    u1[i]=(i+1)/Nx/2.0;
    u2[i]=(i+1)/Nx/4.0;
    y[i]=(i+1)/Nx/6.0;
    w[i]=(i+1)/Nx/8.0;
    for (j=0; j<=Ny-1; j++) {
      A[i][j]=(i*j)/Ny;
      B[i][j]=0;
    }
  }
  for (j=0; j<=Ny-1; j++) {
    v1[j]=(j+1)/Ny/2.0;
    v2[j]=(j+1)/Ny/4.0;
    z[j]=(j+1)/Ny/6.0;
    x[j]=(j+1)/Ny/8.0;
  }
}

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
  init_arrays();

  double annot_t_start=0, annot_t_end=0, annot_t_total=0;
  int annot_i;

  for (annot_i=0; annot_i<REPS; annot_i++)
  {
    annot_t_start = rtclock();




int i,j,ii,jj,it,jt;

for (i=0; i<=min(nx-1,ny-1); i=i+1) {
  x[i]=0;
  w[i]=0;
}
for (i=nx; i<=ny-1; i=i+1) 
  x[i]=0;
for (i=ny; i<=nx-1; i=i+1) 
  w[i]=0;
for (j=0; j<=nx-1; j=j+1) {
  register int cbv_1;
  cbv_1=ny-1;
#pragma ivdep
#pragma vector always
  for (i=0; i<=cbv_1; i=i+1) {
    B[j][i]=u2[j]*v2[i]+u1[j]*v1[i]+A[j][i];
    x[i]=y[j]*B[j][i]+x[i];
  }
}
for (i=0; i<=ny-1; i=i+1) 
  x[i]=z[i]+b*x[i];
{
#pragma omp parallel for
  for (i=0; i<=nx-1; i=i+1) 
    for (j=0; j<=ny-1; j=j+1) 
      w[i]=B[i][j]*x[j]+w[i];
}
for (i=0; i<=nx-1; i=i+1) 
  w[i]=a*w[i];








    annot_t_end = rtclock();
    annot_t_total += annot_t_end - annot_t_start;
  }
  
  annot_t_total = annot_t_total / REPS;

#ifndef TEST
  printf("%f\n", annot_t_total);
#else
  {
    int i, j;
    for (i=0; i<nx; i++) {
      if (i%100==0)
	printf("\n");
      printf("%f ",w[i]);
    }
    printf("\n");
  }
#endif

  return ((int) w[0]); 
}
                                

