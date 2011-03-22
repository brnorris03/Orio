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



 int c1, c2, c3, c4, c5, c6;
 register int lbv, ubv;

if (N >= 3) {
  for (c1=0;c1<=floord(T-1,32);c1++) {
    for (c2=max(0,ceild(16*c1-15,16));c2<=min(floord(T+N-3,32),floord(32*c1+N+29,32));c2++) {
      for (c3=max(max(max(max(0,ceild(64*c2-N-28,32)),ceild(64*c1-29,32)),ceild(16*c2-15,16)),ceild(16*c1+16*c2-15,16));c3<=min(min(min(min(floord(T+N-3,16),floord(64*c2+N+59,32)),floord(32*c2+T+N+28,32)),floord(32*c1+32*c2+N+60,32)),floord(32*c1+N+29,16));c3++) {
        for (c4=max(max(max(max(0,16*c3-N+2),32*c1),32*c2-N+2),-32*c2+32*c3-N-29);c4<=min(min(min(min(T-1,32*c1+31),32*c2+30),-32*c2+32*c3+30),floord(32*c3+29,2));c4++) {

	  /*@ begin Loop(
	    transform Unroll(ufactor=4)
          for (c5=max(max(c4+1,32*c3-c4-N+2),32*c2);c5<=min(min(c4+N-2,32*c2+31),32*c3-c4+30);c5++) {
	    transform UnrollJam(ufactor=4)
            for (c6=max(c4+c5+1,32*c3);c6<=min(c4+c5+N-2,32*c3+31);c6++) {
              A[-c4+c5][-c4-c5+c6]=(A[1+-c4+c5][1+-c4-c5+c6]+A[1+-c4+c5][-c4-c5+c6]+A[1+-c4+c5][-c4-c5+c6-1]+A[-c4+c5][1+-c4-c5+c6]+A[-c4+c5][-c4-c5+c6]+A[-c4+c5][-c4-c5+c6-1]+A[-c4+c5-1][1+-c4-c5+c6]+A[-c4+c5-1][-c4-c5+c6]+A[-c4+c5-1][-c4-c5+c6-1])/9;
            }
          }
	  ) @*/   {
     for (c5=max(max(c4+1,32*c3-c4-N+2),32*c2); c5<=min(min(c4+N-2,32*c2+31),32*c3-c4+30)-3; c5=c5+4) {
       {
         for (c6=max(c4+c5+1,32*c3); c6<=min(c4+c5+N-2,32*c3+31)-3; c6=c6+4) {
           A[-c4+c5][-c4+c6-c5]=0.111111111111*(A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5-1]+A[-c4+c5][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5]+A[-c4+c5][-c4+c6-c5-1]+A[-c4+c5-1][-c4+c6-c5+1]+A[-c4+c5-1][-c4+c6-c5]+A[-c4+c5-1][-c4+c6-c5-1]);
           A[-c4+c5][-c4+c6-c5+1]=0.111111111111*(A[-c4+c5+1][-c4+c6-c5+2]+A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5][-c4+c6-c5+2]+A[-c4+c5][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5]+A[-c4+c5-1][-c4+c6-c5+2]+A[-c4+c5-1][-c4+c6-c5+1]+A[-c4+c5-1][-c4+c6-c5]);
           A[-c4+c5][-c4+c6-c5+2]=0.111111111111*(A[-c4+c5+1][-c4+c6-c5+3]+A[-c4+c5+1][-c4+c6-c5+2]+A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5+3]+A[-c4+c5][-c4+c6-c5+2]+A[-c4+c5][-c4+c6-c5+1]+A[-c4+c5-1][-c4+c6-c5+3]+A[-c4+c5-1][-c4+c6-c5+2]+A[-c4+c5-1][-c4+c6-c5+1]);
           A[-c4+c5][-c4+c6-c5+3]=0.111111111111*(A[-c4+c5+1][-c4+c6-c5+4]+A[-c4+c5+1][-c4+c6-c5+3]+A[-c4+c5+1][-c4+c6-c5+2]+A[-c4+c5][-c4+c6-c5+4]+A[-c4+c5][-c4+c6-c5+3]+A[-c4+c5][-c4+c6-c5+2]+A[-c4+c5-1][-c4+c6-c5+4]+A[-c4+c5-1][-c4+c6-c5+3]+A[-c4+c5-1][-c4+c6-c5+2]);
         }
         for (; c6<=min(c4+c5+N-2,32*c3+31); c6=c6+1) {
           A[-c4+c5][-c4+c6-c5]=0.111111111111*(A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5-1]+A[-c4+c5][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5]+A[-c4+c5][-c4+c6-c5-1]+A[-c4+c5-1][-c4+c6-c5+1]+A[-c4+c5-1][-c4+c6-c5]+A[-c4+c5-1][-c4+c6-c5-1]);
         }
       }
       {
         for (c6=max(c4+c5+2,32*c3); c6<=min(c4+c5+N-1,32*c3+31)-3; c6=c6+4) {
           A[-c4+c5+1][-c4+c6-c5-1]=0.111111111111*(A[-c4+c5+2][-c4+c6-c5]+A[-c4+c5+2][-c4+c6-c5-1]+A[-c4+c5+2][-c4+c6-c5-2]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5-1]+A[-c4+c5+1][-c4+c6-c5-2]+A[-c4+c5][-c4+c6-c5]+A[-c4+c5][-c4+c6-c5-1]+A[-c4+c5][-c4+c6-c5-2]);
           A[-c4+c5+1][-c4+c6-c5]=0.111111111111*(A[-c4+c5+2][-c4+c6-c5+1]+A[-c4+c5+2][-c4+c6-c5]+A[-c4+c5+2][-c4+c6-c5-1]+A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5-1]+A[-c4+c5][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5]+A[-c4+c5][-c4+c6-c5-1]);
           A[-c4+c5+1][-c4+c6-c5+1]=0.111111111111*(A[-c4+c5+2][-c4+c6-c5+2]+A[-c4+c5+2][-c4+c6-c5+1]+A[-c4+c5+2][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5+2]+A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5][-c4+c6-c5+2]+A[-c4+c5][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5]);
           A[-c4+c5+1][-c4+c6-c5+2]=0.111111111111*(A[-c4+c5+2][-c4+c6-c5+3]+A[-c4+c5+2][-c4+c6-c5+2]+A[-c4+c5+2][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5+3]+A[-c4+c5+1][-c4+c6-c5+2]+A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5+3]+A[-c4+c5][-c4+c6-c5+2]+A[-c4+c5][-c4+c6-c5+1]);
         }
         for (; c6<=min(c4+c5+N-1,32*c3+31); c6=c6+1) {
           A[-c4+c5+1][-c4+c6-c5-1]=0.111111111111*(A[-c4+c5+2][-c4+c6-c5]+A[-c4+c5+2][-c4+c6-c5-1]+A[-c4+c5+2][-c4+c6-c5-2]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5-1]+A[-c4+c5+1][-c4+c6-c5-2]+A[-c4+c5][-c4+c6-c5]+A[-c4+c5][-c4+c6-c5-1]+A[-c4+c5][-c4+c6-c5-2]);
         }
       }
       {
         for (c6=max(c4+c5+3,32*c3); c6<=min(c4+c5+N,32*c3+31)-3; c6=c6+4) {
           A[-c4+c5+2][-c4+c6-c5-2]=0.111111111111*(A[-c4+c5+3][-c4+c6-c5-1]+A[-c4+c5+3][-c4+c6-c5-2]+A[-c4+c5+3][-c4+c6-c5-3]+A[-c4+c5+2][-c4+c6-c5-1]+A[-c4+c5+2][-c4+c6-c5-2]+A[-c4+c5+2][-c4+c6-c5-3]+A[-c4+c5+1][-c4+c6-c5-1]+A[-c4+c5+1][-c4+c6-c5-2]+A[-c4+c5+1][-c4+c6-c5-3]);
           A[-c4+c5+2][-c4+c6-c5-1]=0.111111111111*(A[-c4+c5+3][-c4+c6-c5]+A[-c4+c5+3][-c4+c6-c5-1]+A[-c4+c5+3][-c4+c6-c5-2]+A[-c4+c5+2][-c4+c6-c5]+A[-c4+c5+2][-c4+c6-c5-1]+A[-c4+c5+2][-c4+c6-c5-2]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5-1]+A[-c4+c5+1][-c4+c6-c5-2]);
           A[-c4+c5+2][-c4+c6-c5]=0.111111111111*(A[-c4+c5+3][-c4+c6-c5+1]+A[-c4+c5+3][-c4+c6-c5]+A[-c4+c5+3][-c4+c6-c5-1]+A[-c4+c5+2][-c4+c6-c5+1]+A[-c4+c5+2][-c4+c6-c5]+A[-c4+c5+2][-c4+c6-c5-1]+A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5-1]);
           A[-c4+c5+2][-c4+c6-c5+1]=0.111111111111*(A[-c4+c5+3][-c4+c6-c5+2]+A[-c4+c5+3][-c4+c6-c5+1]+A[-c4+c5+3][-c4+c6-c5]+A[-c4+c5+2][-c4+c6-c5+2]+A[-c4+c5+2][-c4+c6-c5+1]+A[-c4+c5+2][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5+2]+A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5]);
         }
         for (; c6<=min(c4+c5+N,32*c3+31); c6=c6+1) {
           A[-c4+c5+2][-c4+c6-c5-2]=0.111111111111*(A[-c4+c5+3][-c4+c6-c5-1]+A[-c4+c5+3][-c4+c6-c5-2]+A[-c4+c5+3][-c4+c6-c5-3]+A[-c4+c5+2][-c4+c6-c5-1]+A[-c4+c5+2][-c4+c6-c5-2]+A[-c4+c5+2][-c4+c6-c5-3]+A[-c4+c5+1][-c4+c6-c5-1]+A[-c4+c5+1][-c4+c6-c5-2]+A[-c4+c5+1][-c4+c6-c5-3]);
         }
       }
       {
         for (c6=max(c4+c5+4,32*c3); c6<=min(c4+c5+N+1,32*c3+31)-3; c6=c6+4) {
           A[-c4+c5+3][-c4+c6-c5-3]=0.111111111111*(A[-c4+c5+4][-c4+c6-c5-2]+A[-c4+c5+4][-c4+c6-c5-3]+A[-c4+c5+4][-c4+c6-c5-4]+A[-c4+c5+3][-c4+c6-c5-2]+A[-c4+c5+3][-c4+c6-c5-3]+A[-c4+c5+3][-c4+c6-c5-4]+A[-c4+c5+2][-c4+c6-c5-2]+A[-c4+c5+2][-c4+c6-c5-3]+A[-c4+c5+2][-c4+c6-c5-4]);
           A[-c4+c5+3][-c4+c6-c5-2]=0.111111111111*(A[-c4+c5+4][-c4+c6-c5-1]+A[-c4+c5+4][-c4+c6-c5-2]+A[-c4+c5+4][-c4+c6-c5-3]+A[-c4+c5+3][-c4+c6-c5-1]+A[-c4+c5+3][-c4+c6-c5-2]+A[-c4+c5+3][-c4+c6-c5-3]+A[-c4+c5+2][-c4+c6-c5-1]+A[-c4+c5+2][-c4+c6-c5-2]+A[-c4+c5+2][-c4+c6-c5-3]);
           A[-c4+c5+3][-c4+c6-c5-1]=0.111111111111*(A[-c4+c5+4][-c4+c6-c5]+A[-c4+c5+4][-c4+c6-c5-1]+A[-c4+c5+4][-c4+c6-c5-2]+A[-c4+c5+3][-c4+c6-c5]+A[-c4+c5+3][-c4+c6-c5-1]+A[-c4+c5+3][-c4+c6-c5-2]+A[-c4+c5+2][-c4+c6-c5]+A[-c4+c5+2][-c4+c6-c5-1]+A[-c4+c5+2][-c4+c6-c5-2]);
           A[-c4+c5+3][-c4+c6-c5]=0.111111111111*(A[-c4+c5+4][-c4+c6-c5+1]+A[-c4+c5+4][-c4+c6-c5]+A[-c4+c5+4][-c4+c6-c5-1]+A[-c4+c5+3][-c4+c6-c5+1]+A[-c4+c5+3][-c4+c6-c5]+A[-c4+c5+3][-c4+c6-c5-1]+A[-c4+c5+2][-c4+c6-c5+1]+A[-c4+c5+2][-c4+c6-c5]+A[-c4+c5+2][-c4+c6-c5-1]);
         }
         for (; c6<=min(c4+c5+N+1,32*c3+31); c6=c6+1) {
           A[-c4+c5+3][-c4+c6-c5-3]=0.111111111111*(A[-c4+c5+4][-c4+c6-c5-2]+A[-c4+c5+4][-c4+c6-c5-3]+A[-c4+c5+4][-c4+c6-c5-4]+A[-c4+c5+3][-c4+c6-c5-2]+A[-c4+c5+3][-c4+c6-c5-3]+A[-c4+c5+3][-c4+c6-c5-4]+A[-c4+c5+2][-c4+c6-c5-2]+A[-c4+c5+2][-c4+c6-c5-3]+A[-c4+c5+2][-c4+c6-c5-4]);
         }
       }
     }
     for (; c5<=min(min(c4+N-2,32*c2+31),32*c3-c4+30); c5=c5+1) {
       {
         for (c6=max(c4+c5+1,32*c3); c6<=min(c4+c5+N-2,32*c3+31)-3; c6=c6+4) {
           A[-c4+c5][-c4+c6-c5]=0.111111111111*(A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5+1][-c4+c6-c5-1]+A[-c4+c5][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5]+A[-c4+c5][-c4+c6-c5-1]+A[-c4+c5-1][-c4+c6-c5+1]+A[-c4+c5-1][-c4+c6-c5]+A[-c4+c5-1][-c4+c6-c5-1]);
           A[-c4+c5][-c4+c6-c5+1]=0.111111111111*(A[-c4+c5+1][-c4+c6-c5+2]+A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5+1][-c4+c6-c5]+A[-c4+c5][-c4+c6-c5+2]+A[-c4+c5][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5]+A[-c4+c5-1][-c4+c6-c5+2]+A[-c4+c5-1][-c4+c6-c5+1]+A[-c4+c5-1][-c4+c6-c5]);
           A[-c4+c5][-c4+c6-c5+2]=0.111111111111*(A[-c4+c5+1][-c4+c6-c5+3]+A[-c4+c5+1][-c4+c6-c5+2]+A[-c4+c5+1][-c4+c6-c5+1]+A[-c4+c5][-c4+c6-c5+3]+A[-c4+c5][-c4+c6-c5+2]+A[-c4+c5][-c4+c6-c5+1]+A[-c4+c5-1][-c4+c6-c5+3]+A[-c4+c5-1][-c4+c6-c5+2]+A[-c4+c5-1][-c4+c6-c5+1]);
           A[-c4+c5][-c4+c6-c5+3]=0.111111111111*(A[-c4+c5+1][-c4+c6-c5+4]+A[-c4+c5+1][-c4+c6-c5+3]+A[-c4+c5+1][-c4+c6-c5+2]+A[-c4+c5][-c4+c6-c5+4]+A[-c4+c5][-c4+c6-c5+3]+A[-c4+c5][-c4+c6-c5+2]+A[-c4+c5-1][-c4+c6-c5+4]+A[-c4+c5-1][-c4+c6-c5+3]+A[-c4+c5-1][-c4+c6-c5+2]);
         }
         for (; c6<=min(c4+c5+N-2,32*c3+31); c6=c6+1) {
           A[-c4+c5][-c4-c5+c6]=(A[1+-c4+c5][1+-c4-c5+c6]+A[1+-c4+c5][-c4-c5+c6]+A[1+-c4+c5][-c4-c5+c6-1]+A[-c4+c5][1+-c4-c5+c6]+A[-c4+c5][-c4-c5+c6]+A[-c4+c5][-c4-c5+c6-1]+A[-c4+c5-1][1+-c4-c5+c6]+A[-c4+c5-1][-c4-c5+c6]+A[-c4+c5-1][-c4-c5+c6-1])/9;
         }
       }
     }
   }
/*@ end @*/

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
                                    
