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
#include <omp.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))



 int c1, c2, c3, c4, c5, c6, c7, c8, c9;
 register int lb, ub, lb1, ub1, lb2, ub2;
 register int lbv, ubv;

if (N >= 2) {
  for (c1=-1;c1<=floord(2*N-3,256);c1++) {
 lb1=max(max(ceild(256*c1-N+2,256),0),ceild(128*c1-127,256));
 ub1=min(floord(256*c1+255,256),floord(N-1,256));
#pragma omp parallel for shared(c1,lb1,ub1) private(c2,c3,c4,c5,c6,c7,c8,c9)
 for (c2=lb1; c2<=ub1; c2++) {
      for (c3=max(ceild(128*c1-128*c2-32385,32640),ceild(128*c1-128*c2-127,128));c3<=floord(N-1,256);c3++) {
        for (c4=max(max(8*c1-8*c2,0),8*c1-8*c2-1792*c3-1778);c4<=min(min(min(min(floord(N-2,32),floord(128*c2+127,16)),floord(3968*c3+3937,16)),8*c1-8*c2+7),floord(128*c3+127,16));c4++) {
          for (c5=max(max(0,ceild(16*c4-15,16)),8*c2);c5<=min(8*c2+7,floord(N-1,32));c5++) {
            for (c6=max(max(max(max(ceild(16*c4-465,496),ceild(8*c1-8*c2-8*c3-c4-217,223)),ceild(-8*c1+8*c2+8*c3+c4-217,225)),8*c3),ceild(16*c4-15,16));c6<=min(floord(N-1,32),8*c3+7);c6++) {
              if ((c1 == c2+c3) && (c4 == c6)) {
                for (c7=max(0,32*c6);c7<=min(min(N-2,32*c5+30),32*c6+30);c7++) {
{
 lbv=max(c7+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
                    {A[c7][c9]=A[c7][c9]/A[c7][c7];} ;
                  }
}
                  for (c8=c7+1;c8<=min(N-1,32*c6+31);c8++) {
{
 lbv=max(32*c5,c7+1); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
                      {A[c8][c9]=A[c8][c9]-A[c8][c7]*A[c7][c9];} ;
                    }
}
                  }
                }
              }
     {
  for (c7 = max(0, 32 * c4); c7 <= min(min(32 * c6 - 1, 32 * c5 + 30), 32 * c4 + 31) - 3; c7 = c7 + 4) {
      for (c8 = 32 * c6; c8 <= min(N - 1, 32 * c6 + 31) - 3; c8 = c8 + 4) {
{
 lbv=max(c7+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
              {A[c8][c9]=A[c8][c9]-A[c8][c7]*A[c7][c9];};
              {A[(c8 + 1)][c9]=A[(c8 + 1)][c9]-A[(c8 + 1)][c7]*A[c7][c9];};
              {A[(c8 + 2)][c9]=A[(c8 + 2)][c9]-A[(c8 + 2)][c7]*A[c7][c9];};
              {A[(c8 + 3)][c9]=A[(c8 + 3)][c9]-A[(c8 + 3)][c7]*A[c7][c9];};
            }
}
{
 lbv=max((c7+1)+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
              {A[c8][c9]=A[c8][c9]-A[c8][(c7 + 1)]*A[(c7 + 1)][c9];};
              {A[(c8 + 1)][c9]=A[(c8 + 1)][c9]-A[(c8 + 1)][(c7 + 1)]*A[(c7 + 1)][c9];};
              {A[(c8 + 2)][c9]=A[(c8 + 2)][c9]-A[(c8 + 2)][(c7 + 1)]*A[(c7 + 1)][c9];};
              {A[(c8 + 3)][c9]=A[(c8 + 3)][c9]-A[(c8 + 3)][(c7 + 1)]*A[(c7 + 1)][c9];};
            }
}
{
 lbv=max((c7+2)+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
              {A[c8][c9]=A[c8][c9]-A[c8][(c7 + 2)]*A[(c7 + 2)][c9];};
              {A[(c8 + 1)][c9]=A[(c8 + 1)][c9]-A[(c8 + 1)][(c7 + 2)]*A[(c7 + 2)][c9];};
              {A[(c8 + 2)][c9]=A[(c8 + 2)][c9]-A[(c8 + 2)][(c7 + 2)]*A[(c7 + 2)][c9];};
              {A[(c8 + 3)][c9]=A[(c8 + 3)][c9]-A[(c8 + 3)][(c7 + 2)]*A[(c7 + 2)][c9];};
            }
}
{
 lbv=max((c7+3)+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
              {A[c8][c9]=A[c8][c9]-A[c8][(c7 + 3)]*A[(c7 + 3)][c9];};
              {A[(c8 + 1)][c9]=A[(c8 + 1)][c9]-A[(c8 + 1)][(c7 + 3)]*A[(c7 + 3)][c9];};
              {A[(c8 + 2)][c9]=A[(c8 + 2)][c9]-A[(c8 + 2)][(c7 + 3)]*A[(c7 + 3)][c9];};
              {A[(c8 + 3)][c9]=A[(c8 + 3)][c9]-A[(c8 + 3)][(c7 + 3)]*A[(c7 + 3)][c9];};
            }
}
        }
      for (; c8 <= min(N - 1, 32 * c6 + 31); c8 = c8 + 1) {
{
 lbv=max(c7+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
              {A[c8][c9]=A[c8][c9]-A[c8][c7]*A[c7][c9];};
            }
}
{
 lbv=max((c7+1)+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
              {A[c8][c9]=A[c8][c9]-A[c8][(c7 + 1)]*A[(c7 + 1)][c9];};
            }
}
{
 lbv=max((c7+2)+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
              {A[c8][c9]=A[c8][c9]-A[c8][(c7 + 2)]*A[(c7 + 2)][c9];};
            }
}
{
 lbv=max((c7+3)+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
              {A[c8][c9]=A[c8][c9]-A[c8][(c7 + 3)]*A[(c7 + 3)][c9];};
            }
}
        }
    }
  for (; c7 <= min(min(32 * c6 - 1, 32 * c5 + 30), 32 * c4 + 31); c7 = c7 + 1) {
      for (c8 = 32 * c6; c8 <= min(N - 1, 32 * c6 + 31) - 3; c8 = c8 + 4)
{
 lbv=max(c7+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
            {A[c8][c9]=A[c8][c9]-A[c8][c7]*A[c7][c9];};
            {A[(c8 + 1)][c9]=A[(c8 + 1)][c9]-A[(c8 + 1)][c7]*A[c7][c9];};
            {A[(c8 + 2)][c9]=A[(c8 + 2)][c9]-A[(c8 + 2)][c7]*A[c7][c9];};
            {A[(c8 + 3)][c9]=A[(c8 + 3)][c9]-A[(c8 + 3)][c7]*A[c7][c9];};
          }
}
      for (; c8 <= min(N - 1, 32 * c6 + 31); c8 = c8 + 1)
{
 lbv=max(c7+1,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
            {A[c8][c9]=A[c8][c9]-A[c8][c7]*A[c7][c9];};
          }
}
    }
}

              if ((c1 == c2+c3) && (-c4 == -c6) && (c4 <= min(floord(N-33,32),floord(32*c5-1,32)))) {
{
 lbv=max(32*c4+32,32*c5); ubv=min(N-1,32*c5+31);
#pragma ivdep
#pragma vector always
 for (c9=lbv; c9<=ubv; c9++) {
                  {A[32*c4+31][c9]=A[32*c4+31][c9]/A[32*c4+31][32*c4+31];} ;
                }
}
              }
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
