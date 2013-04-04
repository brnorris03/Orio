
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#define SITES 2
double *A;
double *x;
double *y;
void malloc_arrays() {
  int i1;
  A = (double*) malloc((18 *SITES) * sizeof(double));
  x = (double*) malloc((6 *SITES) * sizeof(double));
  y = (double*) malloc((6 *SITES) * sizeof(double));
}

void init_input_vars() {
  int i1;
  for (i1=0; i1<18 *SITES; i1++)
   A[i1] = (i1) % 5 + 1;
  for (i1=0; i1<6 *SITES; i1++)
   x[i1] = (i1) % 5 + 1;
  for (i1=0; i1<6 *SITES; i1++)
   y[i1] = 0;
}





extern double getClock(); 

int main(int argc, char *argv[]) {
  malloc_arrays();
  init_input_vars();


  double orio_t_start, orio_t_end, orio_t = (double)LONG_MAX;
  int orio_i;

  for (orio_i=0; orio_i<ORIO_REPS; orio_i++) {
    orio_t_start = getClock();
    
    

  int sites_on_node=SITES;

  /*@ begin Loop(
  transform Composite(scalarreplace = (SREP, 'double'))
  transform RegTile(loops=['j','k'], ufactors=[U1,U2])

  for(i=0; i<=sites_on_node-1; i++) {
    for(j=0; j<=5; j+=2) {
      cr = ci = 0.0;
      for(k=0; k<=5; k+=2) {
        ar=A[18*i+3*j+k];
        ai=A[18*i+3*j+k+1];
        br=x[6*i+k];
        bi=x[6*i+k+1];
        cr += ar*br - ai*bi;
        ci += ar*bi + ai*br;
      }
      y[6*i+j]  =cr;
      y[6*i+j+1]=ci;
    }
  }

  ) @*/
  for (i=0; i<=sites_on_node-1; i++ ) {
    for (j=0; j<=5; j=j+2) {
      cr=ci=0.0;
      for (k=0; k<=5; k=k+2) {
        ar=A[18*i+3*j+k];
        ai=A[18*i+3*j+k+1];
        br=x[6*i+k];
        bi=x[6*i+k+1];
        cr=cr+ar*br-ai*bi;
        ci=ci+ar*bi+ai*br;
      }
      y[6*i+j]=cr;
      y[6*i+j+1]=ci;
    }
  }
/*@ end @*/
  

    orio_t_end = getClock();
    orio_t = orio_t_end - orio_t_start;
    printf("{'[0, 0, 0, 0]' : %g}\n", orio_t);
    if (orio_i==0) {
      
    }
  }
  
  
  return 0;
}
