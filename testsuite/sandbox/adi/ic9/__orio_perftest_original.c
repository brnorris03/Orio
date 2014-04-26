
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#define T 10
#define N 100
double X[N][N +20];
double A[N][N +20];
double B[N][N +20];
void malloc_arrays() {

}

void init_input_vars() {
  int i1,i2;
  for (i1=0; i1<N; i1++)
   for (i2=0; i2<N +20; i2++)
    X[i1][i2] = (i1+i2) % 5 + 1;
  for (i1=0; i1<N; i1++)
   for (i2=0; i2<N +20; i2++)
    A[i1][i2] = (i1+i2) % 5 + 1;
  for (i1=0; i1<N; i1++)
   for (i2=0; i2<N +20; i2++)
    B[i1][i2] = (i1+i2) % 5 + 1;
}





extern double getClock(); 

int main(int argc, char *argv[]) {
  malloc_arrays();
  init_input_vars();


  double orio_t_start, orio_t_end, orio_t = (double)LONG_MAX;
  int orio_i;

  for (orio_i=0; orio_i<ORIO_REPS; orio_i++) {
    orio_t_start = getClock();
    
       

register int i1,i2,t;  

/*@ begin Loop (
 
transform Composite(
scalarreplace = (SCREP, 'double'),
vector = (VEC1, ['ivdep', 'vector always'])
)
transform UnrollJam(ufactor=UF1,parallelize=PAR1)
for (t=0; t<=T-1; t++) 
  {
  transform Composite(
   vector = (VEC2, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=UF2,parallelize=PAR2)
  for (i1=0; i1<=N-1; i1++) 
    transform Composite(
    vector = (VEC3, ['ivdep', 'vector always'])
    )
    transform UnrollJam(ufactor=UF3,parallelize=PAR3)
    for (i2=1; i2<=N-1; i2++) 
    {
     X[i1][i2] = X[i1][i2] - X[i1][i2-1] * A[i1][i2] / B[i1][i2-1];
     B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2-1];
     }
       
   transform Composite(
   permut = [PERM2],
   scalarreplace = (SCREP, 'double'),
   regtile = (['i1','i2'],[T2_1,T2_2]),
   vector = (VEC4, ['ivdep', 'vector always'])
   )
   transform UnrollJam(ufactor=UF4,parallelize=PAR4)
   for (i1=1; i1<=N-1; i1++) 
     transform Composite(
     vector = (VEC5, ['ivdep', 'vector always'])
      )
      transform UnrollJam(ufactor=UF5,parallelize=PAR5) 
      for (i2=0; i2<=N-1; i2++) 
      {
      X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2];
      B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2];
       }
  }

) @*/
{
  int t;
  for (t=0; t<=T-1; t=t+1) {
    {
      int i1;
      for (i1=0; i1<=N-1; i1=i1+1) {
        int i2;
        for (i2=1; i2<=N-1; i2=i2+1) {
          X[i1][i2]=X[i1][i2]-X[i1][i2-1]*A[i1][i2]/B[i1][i2-1];
          B[i1][i2]=B[i1][i2]-A[i1][i2]*A[i1][i2]/B[i1][i2-1];
        }
      }
    }
    {
      int i1;
      for (i1=1; i1<=N-1; i1=i1+1) {
        int i2;
        for (i2=0; i2<=N-1; i2=i2+1) {
          X[i1][i2]=X[i1][i2]-X[i1-1][i2]*A[i1][i2]/B[i1-1][i2];
          B[i1][i2]=B[i1][i2]-A[i1][i2]*A[i1][i2]/B[i1-1][i2];
        }
      }
    }
  }
}
/*@ end @*/


    orio_t_end = getClock();
    orio_t = orio_t_end - orio_t_start;
    printf("{'[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' : %g}\n", orio_t);
    if (orio_i==0) {
      
    }
  }
  
  
  return 0;
}
