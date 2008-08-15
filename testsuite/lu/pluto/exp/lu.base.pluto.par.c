#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

double L[N][N];
double U[N][N];
double A[N][N+13];

void print_array()
{
  int i, j;

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      fprintf(stderr, "%lf ", round(A[i][j]));
      if (j%80 == 79) fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

void init_arrays()
{
  int i, j, k;

  /* have to initialize this matrix properly to prevent                                              
   * division by zero                                                                                 
   */
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      L[i][j] = 0.0;
      U[i][j] = 0.0;
    }
  }

  for (i=0; i<N; i++) {
    for (j=0; j<=i; j++) {
      L[i][j] = i+j+1;
      U[j][i] = i+j+1;
    }
  }

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      for (k=0; k<N; k++) {
	A[i][j] += L[i][k]*U[k][j];
      }
    }
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




#include <math.h>
#include <assert.h>

#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

	#define S1(zT0,zT1,k,j)	{A[k][j]=A[k][j]/A[k][k];}
	#define S2(zT0,zT1,zT2,k,i,j)	{A[i][j]=A[i][j]-A[i][k]*A[k][j];}

	int c1, c2, c3, c4, c5, c6;

	register int lb, ub, lb1, ub1, lb2, ub2;
	register int lbv, ubv;

/* Generated from PLuTo-produced CLooG file by CLooG v0.14.1 64 bits in 0.02s. */
for (c1=-1;c1<=floord(2*N-3,32);c1++) {
	lb1=max(max(ceild(16*c1-15,32),ceild(32*c1-N+2,32)),0);
	ub1=min(floord(32*c1+31,32),floord(N-1,32));
#pragma omp parallel for shared(c1,lb1,ub1) private(c2,c3,c4,c5,c6)
	for (c2=lb1; c2<=ub1; c2++) {
    for (c3=max(ceild(16*c1-16*c2-465,496),ceild(16*c1-16*c2-15,16));c3<=floord(N-1,32);c3++) {
      if (c1 == c2+c3) {
        for (c4=max(0,32*c3);c4<=min(min(32*c3+30,32*c2+30),N-2);c4++) {
{
	lbv=max(c4+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
            S1(c1-c2,c2,c4,c6) ;
          }
}
          for (c5=c4+1;c5<=min(32*c3+31,N-1);c5++) {
{
	lbv=max(c4+1,32*c2);
	ubv=min(32*c2+31,N-1);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1-c2,c1-c2,c2,c4,c5,c6) ;
            }
}
          }
        }
      }
/*@ begin Loop(
transform UnrollJam(ufactor=8)
      for (c4=max(0,32*c1-32*c2);c4<=min(min(32*c1-32*c2+31,32*c3-1),32*c2+30);c4++) 
transform UnrollJam(ufactor=8)
        for (c5=32*c3;c5<=min(N-1,32*c3+31);c5++) 
{
{
	lbv=max(c4+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
            S2(c1-c2,c3,c2,c4,c5,c6) ;
          }
}
}
) @*/{ 

  for (c4 = max(0, 32 * c1 - 32 * c2); c4 <= min(min(32 * c1 - 32 * c2 + 31, 32 * c3 - 1), 32 * c2 + 30) - 7; c4 = c4 + 8)     { 

      for (c5 = 32 * c3; c5 <= min(N - 1, 32 * c3 + 31) - 7; c5 = c5 + 8)         { 

{
	lbv=max(c4+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, c4, c5, c6); 
              S2(c1 - c2, c3, c2, c4, (c5 + 1), c6); 
              S2(c1 - c2, c3, c2, c4, (c5 + 2), c6); 
              S2(c1 - c2, c3, c2, c4, (c5 + 3), c6); 
              S2(c1 - c2, c3, c2, c4, (c5 + 4), c6); 
              S2(c1 - c2, c3, c2, c4, (c5 + 5), c6); 
              S2(c1 - c2, c3, c2, c4, (c5 + 6), c6); 
              S2(c1 - c2, c3, c2, c4, (c5 + 7), c6); 
            } 
}

{
	lbv=max((c4+1)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 1), c5, c6); 
              S2(c1 - c2, c3, c2, (c4 + 1), (c5 + 1), c6); 
              S2(c1 - c2, c3, c2, (c4 + 1), (c5 + 2), c6); 
              S2(c1 - c2, c3, c2, (c4 + 1), (c5 + 3), c6); 
              S2(c1 - c2, c3, c2, (c4 + 1), (c5 + 4), c6); 
              S2(c1 - c2, c3, c2, (c4 + 1), (c5 + 5), c6); 
              S2(c1 - c2, c3, c2, (c4 + 1), (c5 + 6), c6); 
              S2(c1 - c2, c3, c2, (c4 + 1), (c5 + 7), c6); 
            } 
}

{
	lbv=max((c4+2)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 2), c5, c6); 
              S2(c1 - c2, c3, c2, (c4 + 2), (c5 + 1), c6); 
              S2(c1 - c2, c3, c2, (c4 + 2), (c5 + 2), c6); 
              S2(c1 - c2, c3, c2, (c4 + 2), (c5 + 3), c6); 
              S2(c1 - c2, c3, c2, (c4 + 2), (c5 + 4), c6); 
              S2(c1 - c2, c3, c2, (c4 + 2), (c5 + 5), c6); 
              S2(c1 - c2, c3, c2, (c4 + 2), (c5 + 6), c6); 
              S2(c1 - c2, c3, c2, (c4 + 2), (c5 + 7), c6); 
            } 
}

{
	lbv=max((c4+3)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 3), c5, c6); 
              S2(c1 - c2, c3, c2, (c4 + 3), (c5 + 1), c6); 
              S2(c1 - c2, c3, c2, (c4 + 3), (c5 + 2), c6); 
              S2(c1 - c2, c3, c2, (c4 + 3), (c5 + 3), c6); 
              S2(c1 - c2, c3, c2, (c4 + 3), (c5 + 4), c6); 
              S2(c1 - c2, c3, c2, (c4 + 3), (c5 + 5), c6); 
              S2(c1 - c2, c3, c2, (c4 + 3), (c5 + 6), c6); 
              S2(c1 - c2, c3, c2, (c4 + 3), (c5 + 7), c6); 
            } 
}

{
	lbv=max((c4+4)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 4), c5, c6); 
              S2(c1 - c2, c3, c2, (c4 + 4), (c5 + 1), c6); 
              S2(c1 - c2, c3, c2, (c4 + 4), (c5 + 2), c6); 
              S2(c1 - c2, c3, c2, (c4 + 4), (c5 + 3), c6); 
              S2(c1 - c2, c3, c2, (c4 + 4), (c5 + 4), c6); 
              S2(c1 - c2, c3, c2, (c4 + 4), (c5 + 5), c6); 
              S2(c1 - c2, c3, c2, (c4 + 4), (c5 + 6), c6); 
              S2(c1 - c2, c3, c2, (c4 + 4), (c5 + 7), c6); 
            } 
}

{
	lbv=max((c4+5)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 5), c5, c6); 
              S2(c1 - c2, c3, c2, (c4 + 5), (c5 + 1), c6); 
              S2(c1 - c2, c3, c2, (c4 + 5), (c5 + 2), c6); 
              S2(c1 - c2, c3, c2, (c4 + 5), (c5 + 3), c6); 
              S2(c1 - c2, c3, c2, (c4 + 5), (c5 + 4), c6); 
              S2(c1 - c2, c3, c2, (c4 + 5), (c5 + 5), c6); 
              S2(c1 - c2, c3, c2, (c4 + 5), (c5 + 6), c6); 
              S2(c1 - c2, c3, c2, (c4 + 5), (c5 + 7), c6); 
            } 
}

{
	lbv=max((c4+6)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 6), c5, c6); 
              S2(c1 - c2, c3, c2, (c4 + 6), (c5 + 1), c6); 
              S2(c1 - c2, c3, c2, (c4 + 6), (c5 + 2), c6); 
              S2(c1 - c2, c3, c2, (c4 + 6), (c5 + 3), c6); 
              S2(c1 - c2, c3, c2, (c4 + 6), (c5 + 4), c6); 
              S2(c1 - c2, c3, c2, (c4 + 6), (c5 + 5), c6); 
              S2(c1 - c2, c3, c2, (c4 + 6), (c5 + 6), c6); 
              S2(c1 - c2, c3, c2, (c4 + 6), (c5 + 7), c6); 
            } 
}

{
	lbv=max((c4+7)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 7), c5, c6); 
              S2(c1 - c2, c3, c2, (c4 + 7), (c5 + 1), c6); 
              S2(c1 - c2, c3, c2, (c4 + 7), (c5 + 2), c6); 
              S2(c1 - c2, c3, c2, (c4 + 7), (c5 + 3), c6); 
              S2(c1 - c2, c3, c2, (c4 + 7), (c5 + 4), c6); 
              S2(c1 - c2, c3, c2, (c4 + 7), (c5 + 5), c6); 
              S2(c1 - c2, c3, c2, (c4 + 7), (c5 + 6), c6); 
              S2(c1 - c2, c3, c2, (c4 + 7), (c5 + 7), c6); 
            } 
}
        } 

      for (; c5 <= min(N - 1, 32 * c3 + 31); c5 = c5 + 1)         { 

{
	lbv=max(c4+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, c4, c5, c6); 
            } 
}

{
	lbv=max((c4+1)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 1), c5, c6); 
            } 
}

{
	lbv=max((c4+2)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 2), c5, c6); 
            } 
}

{
	lbv=max((c4+3)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 3), c5, c6); 
            } 
}

{
	lbv=max((c4+4)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 4), c5, c6); 
            } 
}

{
	lbv=max((c4+5)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 5), c5, c6); 
            } 
}

{
	lbv=max((c4+6)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 6), c5, c6); 
            } 
}

{
	lbv=max((c4+7)+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
              S2(c1 - c2, c3, c2, (c4 + 7), c5, c6); 
            } 
}
        } 
    } 

  for (; c4 <= min(min(32 * c1 - 32 * c2 + 31, 32 * c3 - 1), 32 * c2 + 30); c4 = c4 + 1)     { 

      for (c5 = 32 * c3; c5 <= min(N - 1, 32 * c3 + 31) - 7; c5 = c5 + 8) 
{
	lbv=max(c4+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
            S2(c1 - c2, c3, c2, c4, c5, c6); 
            S2(c1 - c2, c3, c2, c4, (c5 + 1), c6); 
            S2(c1 - c2, c3, c2, c4, (c5 + 2), c6); 
            S2(c1 - c2, c3, c2, c4, (c5 + 3), c6); 
            S2(c1 - c2, c3, c2, c4, (c5 + 4), c6); 
            S2(c1 - c2, c3, c2, c4, (c5 + 5), c6); 
            S2(c1 - c2, c3, c2, c4, (c5 + 6), c6); 
            S2(c1 - c2, c3, c2, c4, (c5 + 7), c6); 
          } 
}

      for (; c5 <= min(N - 1, 32 * c3 + 31); c5 = c5 + 1) 
{
	lbv=max(c4+1,32*c2);
	ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
            S2(c1 - c2, c3, c2, c4, c5, c6); 
          } 
}
    } 
} 
/*@ end @*/
      if ((-c1 == -c2-c3) && (c1 <= min(floord(64*c2-1,32),floord(32*c2+N-33,32)))) {
{
	lbv=max(32*c1-32*c2+32,32*c2);
	ubv=min(32*c2+31,N-1);
#pragma ivdep
#pragma vector always
	for (c6=lbv; c6<=ubv; c6++) {
          S1(c1-c2,c2,32*c1-32*c2+31,c6) ;
        }
}
      }
    }
  }
}
/* End of CLooG code */




      annot_t_end = rtclock();
      annot_t_total += annot_t_end - annot_t_start;
    }

  annot_t_total = annot_t_total / REPS;

#ifndef TEST
  printf("%f\n", annot_t_total);
#else
  {
    int i, j;
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        if (j%100==0)
          printf("\n");
        printf("%f ",A[i][j]);
      }
      printf("\n");
    }
  }
#endif

  return ((int) A[0][0]);

}
