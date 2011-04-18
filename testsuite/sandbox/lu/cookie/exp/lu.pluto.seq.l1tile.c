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



 int c1, c2, c3, c4, c5, c6;
 register int lbv, ubv;

if (N >= 2) {
  for (c1=0;c1<=floord(N-2,32);c1++) {
    for (c2=max(ceild(16*c1-15,16),0);c2<=floord(N-1,32);c2++) {
      for (c3=max(ceild(16*c1-465,496),ceild(16*c1-15,16));c3<=floord(N-1,32);c3++) {
        if (c1 == c3) {
          for (c4=max(0,32*c3);c4<=min(min(N-2,32*c3+30),32*c2+30);c4++) {
{
 lbv=max(c4+1,32*c2); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
              {A[c4][c6]=A[c4][c6]/A[c4][c4];} ;
            }
}
            for (c5=c4+1;c5<=min(32*c3+31,N-1);c5++) {
{
 lbv=max(c4+1,32*c2); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
                {A[c5][c6]=A[c5][c6]-A[c5][c4]*A[c4][c6];} ;
              }
}
            }
          }
        }
     {
  for (c4 = max(0, 32 * c1); c4 <= min(min(32 * c3 - 1, 32 * c1 + 31), 32 * c2 + 30) - 3; c4 = c4 + 4) {
      for (c5 = 32 * c3; c5 <= min(N - 1, 32 * c3 + 31) - 3; c5 = c5 + 4) {
{
 lbv=max(32*c2,c4+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
              {A[c5][c6]=A[c5][c6]-A[c5][c4]*A[c4][c6];};
              {A[(c5 + 1)][c6]=A[(c5 + 1)][c6]-A[(c5 + 1)][c4]*A[c4][c6];};
              {A[(c5 + 2)][c6]=A[(c5 + 2)][c6]-A[(c5 + 2)][c4]*A[c4][c6];};
              {A[(c5 + 3)][c6]=A[(c5 + 3)][c6]-A[(c5 + 3)][c4]*A[c4][c6];};
            }
}
{
 lbv=max(32*c2,(c4+1)+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
              {A[c5][c6]=A[c5][c6]-A[c5][(c4 + 1)]*A[(c4 + 1)][c6];};
              {A[(c5 + 1)][c6]=A[(c5 + 1)][c6]-A[(c5 + 1)][(c4 + 1)]*A[(c4 + 1)][c6];};
              {A[(c5 + 2)][c6]=A[(c5 + 2)][c6]-A[(c5 + 2)][(c4 + 1)]*A[(c4 + 1)][c6];};
              {A[(c5 + 3)][c6]=A[(c5 + 3)][c6]-A[(c5 + 3)][(c4 + 1)]*A[(c4 + 1)][c6];};
            }
}
{
 lbv=max(32*c2,(c4+2)+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
              {A[c5][c6]=A[c5][c6]-A[c5][(c4 + 2)]*A[(c4 + 2)][c6];};
              {A[(c5 + 1)][c6]=A[(c5 + 1)][c6]-A[(c5 + 1)][(c4 + 2)]*A[(c4 + 2)][c6];};
              {A[(c5 + 2)][c6]=A[(c5 + 2)][c6]-A[(c5 + 2)][(c4 + 2)]*A[(c4 + 2)][c6];};
              {A[(c5 + 3)][c6]=A[(c5 + 3)][c6]-A[(c5 + 3)][(c4 + 2)]*A[(c4 + 2)][c6];};
            }
}
{
 lbv=max(32*c2,(c4+3)+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
              {A[c5][c6]=A[c5][c6]-A[c5][(c4 + 3)]*A[(c4 + 3)][c6];};
              {A[(c5 + 1)][c6]=A[(c5 + 1)][c6]-A[(c5 + 1)][(c4 + 3)]*A[(c4 + 3)][c6];};
              {A[(c5 + 2)][c6]=A[(c5 + 2)][c6]-A[(c5 + 2)][(c4 + 3)]*A[(c4 + 3)][c6];};
              {A[(c5 + 3)][c6]=A[(c5 + 3)][c6]-A[(c5 + 3)][(c4 + 3)]*A[(c4 + 3)][c6];};
            }
}
        }
      for (; c5 <= min(N - 1, 32 * c3 + 31); c5 = c5 + 1) {
{
 lbv=max(32*c2,c4+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
              {A[c5][c6]=A[c5][c6]-A[c5][c4]*A[c4][c6];};
            }
}
{
 lbv=max(32*c2,(c4+1)+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
              {A[c5][c6]=A[c5][c6]-A[c5][(c4 + 1)]*A[(c4 + 1)][c6];};
            }
}
{
 lbv=max(32*c2,(c4+2)+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
              {A[c5][c6]=A[c5][c6]-A[c5][(c4 + 2)]*A[(c4 + 2)][c6];};
            }
}
{
 lbv=max(32*c2,(c4+3)+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
              {A[c5][c6]=A[c5][c6]-A[c5][(c4 + 3)]*A[(c4 + 3)][c6];};
            }
}
        }
    }
  for (; c4 <= min(min(32 * c3 - 1, 32 * c1 + 31), 32 * c2 + 30); c4 = c4 + 1) {
      for (c5 = 32 * c3; c5 <= min(N - 1, 32 * c3 + 31) - 3; c5 = c5 + 4)
{
 lbv=max(32*c2,c4+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
            {A[c5][c6]=A[c5][c6]-A[c5][c4]*A[c4][c6];};
            {A[(c5 + 1)][c6]=A[(c5 + 1)][c6]-A[(c5 + 1)][c4]*A[c4][c6];};
            {A[(c5 + 2)][c6]=A[(c5 + 2)][c6]-A[(c5 + 2)][c4]*A[c4][c6];};
            {A[(c5 + 3)][c6]=A[(c5 + 3)][c6]-A[(c5 + 3)][c4]*A[c4][c6];};
          }
}
      for (; c5 <= min(N - 1, 32 * c3 + 31); c5 = c5 + 1)
{
 lbv=max(32*c2,c4+1); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
            {A[c5][c6]=A[c5][c6]-A[c5][c4]*A[c4][c6];};
          }
}
    }
}

        if ((-c1 == -c3) && (c1 <= min(floord(N-33,32),floord(32*c2-1,32)))) {
{
 lbv=max(32*c1+32,32*c2); ubv=min(N-1,32*c2+31);
#pragma ivdep
#pragma vector always
 for (c6=lbv; c6<=ubv; c6++) {
            {A[32*c1+31][c6]=A[32*c1+31][c6]/A[32*c1+31][32*c1+31];} ;
          }
}
        }
      }
    }
  }
}




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
