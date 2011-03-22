#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

double A[N][N+13];

void init_arrays()
{
  int i, j;
  for (i=0; i<N; i++) 
    for (j=0; j<N; j++) 
      A[i][j] = i*i+j*j;
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

	#define S1(zT0,zT1,zT2,t,i,j)	{A[i][j]=(A[1+i][1+j]+A[1+i][j]+A[1+i][j-1]+A[i][1+j]+A[i][j]+A[i][j-1]+A[i-1][1+j]+A[i-1][j]+A[i-1][j-1])/9;}

	int c1, c2, c3, c4, c5, c6;

	register int lb, ub, lb1, ub1, lb2, ub2;
	register int lbv, ubv;

/* Generated from PLuTo-produced CLooG file by CLooG v0.14.1 64 bits in 0.02s. */
for (c1=-1;c1<=floord(2*T+N-4,32);c1++) {
	lb1=max(max(0,ceild(16*c1-15,32)),ceild(32*c1-T+1,32));
	ub1=min(min(floord(T+N-3,32),floord(32*c1+31,32)),floord(32*c1+N+29,64));
#pragma omp parallel for shared(c1,lb1,ub1) private(c2,c3,c4,c5,c6)
	for (c2=lb1; c2<=ub1; c2++) {
    for (c3=max(max(max(max(ceild(64*c2-N-28,32),0),ceild(16*c2-15,16)),ceild(16*c1-15,16)),ceild(64*c1-64*c2-29,32));c3<=min(min(min(min(floord(32*c1-32*c2+N+29,16),floord(T+N-3,16)),floord(32*c2+T+N+28,32)),floord(64*c2+N+59,32)),floord(32*c1+N+60,32));c3++) {
      for (c4=max(max(max(max(-32*c2+32*c3-N-29,16*c3-N+2),32*c2-N+2),0),32*c1-32*c2);c4<=min(min(min(min(32*c1-32*c2+31,T-1),floord(32*c3+29,2)),32*c2+30),-32*c2+32*c3+30);c4++) {
/*@ begin Loop(
transform UnrollJam(ufactor=8)
        for (c5=max(max(32*c2,32*c3-c4-N+2),c4+1);c5<=min(min(c4+N-2,32*c2+31),32*c3-c4+30);c5++) 
transform Unroll(ufactor=8)
          for (c6=max(c4+c5+1,32*c3);c6<=min(c4+c5+N-2,32*c3+31);c6++) 
{
            S1(c1-c2,-c1+2*c2,-c1+c3,c4,-c4+c5,-c4-c5+c6) ;
}
) @*/{ 

  for (c5 = max(max(32 * c2, 32 * c3 - c4 - N + 2), c4 + 1); c5 <= min(min(c4 + N - 2, 32 * c2 + 31), 32 * c3 - c4 + 30) - 7; c5 = c5 + 8)     { 

      for (c6 = max(c4 + c5 + 1, 32 * c3); c6 <= min(c4 + c5 + N - 2, 32 * c3 + 31) - 7; c6 = c6 + 8)         { 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + c6); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 1)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 2)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 3)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 4)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 5)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 6)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 7)); 
        } 

      for (; c6 <= min(c4 + c5 + N - 2, 32 * c3 + 31); c6 = c6 + 1)         S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + c6); 

      for (c6 = max(c4 + (c5 + 1) + 1, 32 * c3); c6 <= min(c4 + (c5 + 1) + N - 2, 32 * c3 + 31) - 7; c6 = c6 + 8)         { 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 1), -c4 - (c5 + 1) + c6); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 1), -c4 - (c5 + 1) + (c6 + 1)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 1), -c4 - (c5 + 1) + (c6 + 2)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 1), -c4 - (c5 + 1) + (c6 + 3)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 1), -c4 - (c5 + 1) + (c6 + 4)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 1), -c4 - (c5 + 1) + (c6 + 5)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 1), -c4 - (c5 + 1) + (c6 + 6)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 1), -c4 - (c5 + 1) + (c6 + 7)); 
        } 

      for (; c6 <= min(c4 + (c5 + 1) + N - 2, 32 * c3 + 31); c6 = c6 + 1)         S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 1), -c4 - (c5 + 1) + c6); 

      for (c6 = max(c4 + (c5 + 2) + 1, 32 * c3); c6 <= min(c4 + (c5 + 2) + N - 2, 32 * c3 + 31) - 7; c6 = c6 + 8)         { 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 2), -c4 - (c5 + 2) + c6); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 2), -c4 - (c5 + 2) + (c6 + 1)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 2), -c4 - (c5 + 2) + (c6 + 2)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 2), -c4 - (c5 + 2) + (c6 + 3)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 2), -c4 - (c5 + 2) + (c6 + 4)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 2), -c4 - (c5 + 2) + (c6 + 5)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 2), -c4 - (c5 + 2) + (c6 + 6)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 2), -c4 - (c5 + 2) + (c6 + 7)); 
        } 

      for (; c6 <= min(c4 + (c5 + 2) + N - 2, 32 * c3 + 31); c6 = c6 + 1)         S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 2), -c4 - (c5 + 2) + c6); 

      for (c6 = max(c4 + (c5 + 3) + 1, 32 * c3); c6 <= min(c4 + (c5 + 3) + N - 2, 32 * c3 + 31) - 7; c6 = c6 + 8)         { 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 3), -c4 - (c5 + 3) + c6); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 3), -c4 - (c5 + 3) + (c6 + 1)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 3), -c4 - (c5 + 3) + (c6 + 2)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 3), -c4 - (c5 + 3) + (c6 + 3)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 3), -c4 - (c5 + 3) + (c6 + 4)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 3), -c4 - (c5 + 3) + (c6 + 5)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 3), -c4 - (c5 + 3) + (c6 + 6)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 3), -c4 - (c5 + 3) + (c6 + 7)); 
        } 

      for (; c6 <= min(c4 + (c5 + 3) + N - 2, 32 * c3 + 31); c6 = c6 + 1)         S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 3), -c4 - (c5 + 3) + c6); 

      for (c6 = max(c4 + (c5 + 4) + 1, 32 * c3); c6 <= min(c4 + (c5 + 4) + N - 2, 32 * c3 + 31) - 7; c6 = c6 + 8)         { 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 4), -c4 - (c5 + 4) + c6); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 4), -c4 - (c5 + 4) + (c6 + 1)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 4), -c4 - (c5 + 4) + (c6 + 2)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 4), -c4 - (c5 + 4) + (c6 + 3)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 4), -c4 - (c5 + 4) + (c6 + 4)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 4), -c4 - (c5 + 4) + (c6 + 5)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 4), -c4 - (c5 + 4) + (c6 + 6)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 4), -c4 - (c5 + 4) + (c6 + 7)); 
        } 

      for (; c6 <= min(c4 + (c5 + 4) + N - 2, 32 * c3 + 31); c6 = c6 + 1)         S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 4), -c4 - (c5 + 4) + c6); 

      for (c6 = max(c4 + (c5 + 5) + 1, 32 * c3); c6 <= min(c4 + (c5 + 5) + N - 2, 32 * c3 + 31) - 7; c6 = c6 + 8)         { 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 5), -c4 - (c5 + 5) + c6); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 5), -c4 - (c5 + 5) + (c6 + 1)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 5), -c4 - (c5 + 5) + (c6 + 2)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 5), -c4 - (c5 + 5) + (c6 + 3)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 5), -c4 - (c5 + 5) + (c6 + 4)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 5), -c4 - (c5 + 5) + (c6 + 5)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 5), -c4 - (c5 + 5) + (c6 + 6)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 5), -c4 - (c5 + 5) + (c6 + 7)); 
        } 

      for (; c6 <= min(c4 + (c5 + 5) + N - 2, 32 * c3 + 31); c6 = c6 + 1)         S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 5), -c4 - (c5 + 5) + c6); 

      for (c6 = max(c4 + (c5 + 6) + 1, 32 * c3); c6 <= min(c4 + (c5 + 6) + N - 2, 32 * c3 + 31) - 7; c6 = c6 + 8)         { 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 6), -c4 - (c5 + 6) + c6); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 6), -c4 - (c5 + 6) + (c6 + 1)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 6), -c4 - (c5 + 6) + (c6 + 2)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 6), -c4 - (c5 + 6) + (c6 + 3)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 6), -c4 - (c5 + 6) + (c6 + 4)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 6), -c4 - (c5 + 6) + (c6 + 5)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 6), -c4 - (c5 + 6) + (c6 + 6)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 6), -c4 - (c5 + 6) + (c6 + 7)); 
        } 


      for (; c6 <= min(c4 + (c5 + 6) + N - 2, 32 * c3 + 31); c6 = c6 + 1)         S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 6), -c4 - (c5 + 6) + c6); 

      for (c6 = max(c4 + (c5 + 7) + 1, 32 * c3); c6 <= min(c4 + (c5 + 7) + N - 2, 32 * c3 + 31) - 7; c6 = c6 + 8)         { 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 7), -c4 - (c5 + 7) + c6); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 7), -c4 - (c5 + 7) + (c6 + 1)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 7), -c4 - (c5 + 7) + (c6 + 2)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 7), -c4 - (c5 + 7) + (c6 + 3)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 7), -c4 - (c5 + 7) + (c6 + 4)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 7), -c4 - (c5 + 7) + (c6 + 5)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 7), -c4 - (c5 + 7) + (c6 + 6)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 7), -c4 - (c5 + 7) + (c6 + 7)); 
        } 



      for (; c6 <= min(c4 + (c5 + 7) + N - 2, 32 * c3 + 31); c6 = c6 + 1)         S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + (c5 + 7), -c4 - (c5 + 7) + c6); 
    } 

  for (; c5 <= min(min(c4 + N - 2, 32 * c2 + 31), 32 * c3 - c4 + 30); c5 = c5 + 1)     { 

      for (c6 = max(c4 + c5 + 1, 32 * c3); c6 <= min(c4 + c5 + N - 2, 32 * c3 + 31) - 7; c6 = c6 + 8)         { 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + c6); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 1)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 2)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 3)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 4)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 5)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 6)); 
          S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + (c6 + 7)); 
        } 

      for (; c6 <= min(c4 + c5 + N - 2, 32 * c3 + 31); c6 = c6 + 1)         S1(c1 - c2, -c1 + 2 * c2, -c1 + c3, c4, -c4 + c5, -c4 - c5 + c6); 
    } 
} 
/*@ end @*/
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
                                    


