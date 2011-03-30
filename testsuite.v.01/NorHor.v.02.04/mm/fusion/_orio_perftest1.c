
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#define CONT 1000
#define NCONT 100
#define M 100
#define N 100
#define K 1000
double A[M][K];
double B[K][N];
double C[M][N];
void malloc_arrays() {

}

void init_input_vars() {
  int i1,i2;
  for (i1=0; i1<M; i1++)
   for (i2=0; i2<K; i2++)
    A[i1][i2] = (i1+i2) % 5 + 1;
  for (i1=0; i1<K; i1++)
   for (i2=0; i2<N; i2++)
    B[i1][i2] = (i1+i2) % 5 + 1;
  for (i1=0; i1<M; i1++)
   for (i2=0; i2<N; i2++)
    C[i1][i2] = 0;
}



extern double getClock(); 

int main(int argc, char *argv[])
{
  malloc_arrays();
init_input_vars();


  double orio_t_start, orio_t_end, orio_t, orio_t_min = (double)LONG_MAX;
  double orio_times[ORIO_TIMES_ARRAY_SIZE];
  int orio_i;

  for (orio_i=0; orio_i<ORIO_REPS; orio_i++)
  {
    orio_t_start = getClock();
    
    

int i, j, k;
int ii, jj, kk;
int iii, jjj, kkk;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(
  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),('k',T1_K,'kk'),
            (('ii','i'),T1_Ia,'iii'),(('jj','j'),T1_Ja,'jjj'),(('kk','k'),T1_Ka,'kkk')],
    permut = [PERMUT],
    unrolljam = (['i','j','k'],[U_I,U_J,U_K]),
    scalarreplace = (SCREP, 'double', 'scv_'),
    vector = (VEC, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
  )
  for(i=0; i<=M-1; i++) 
    for(j=0; j<=N-1; j++)   
      for(k=0; k<=K-1; k++) 
        C[i][j] = C[i][j] + A[i][k] * B[k][j]; 

) @*/
for (jjj=0; jjj<=N-1; jjj=jjj+128) 
  for (kkk=0; kkk<=K-1; kkk=kkk+256) 
    for (iii=0; iii<=M-1; iii=iii+512) 
      for (jj=jjj; jj<=min(N-1,jjj+112); jj=jj+16) 
        for (kk=kkk; kk<=min(K-1,kkk+240); kk=kk+16) {
          for (i=iii; i<=min(M-1,iii+511)-22; i=i+23) {
            for (k=kk; k<=min(K-1,kk+15); k=k+1) {
              register int cbv_1;
              cbv_1=min(N-1,jj+15)-2;
#pragma ivdep
#pragma vector always
              for (j=jj; j<=cbv_1; j=j+3) {
                C[i][j]=C[i][j]+A[i][k]*B[k][j];
                C[i][(j+1)]=C[i][(j+1)]+A[i][k]*B[k][(j+1)];
                C[i][(j+2)]=C[i][(j+2)]+A[i][k]*B[k][(j+2)];
                C[(i+1)][j]=C[(i+1)][j]+A[(i+1)][k]*B[k][j];
                C[(i+1)][(j+1)]=C[(i+1)][(j+1)]+A[(i+1)][k]*B[k][(j+1)];
                C[(i+1)][(j+2)]=C[(i+1)][(j+2)]+A[(i+1)][k]*B[k][(j+2)];
                C[(i+2)][j]=C[(i+2)][j]+A[(i+2)][k]*B[k][j];
                C[(i+2)][(j+1)]=C[(i+2)][(j+1)]+A[(i+2)][k]*B[k][(j+1)];
                C[(i+2)][(j+2)]=C[(i+2)][(j+2)]+A[(i+2)][k]*B[k][(j+2)];
                C[(i+3)][j]=C[(i+3)][j]+A[(i+3)][k]*B[k][j];
                C[(i+3)][(j+1)]=C[(i+3)][(j+1)]+A[(i+3)][k]*B[k][(j+1)];
                C[(i+3)][(j+2)]=C[(i+3)][(j+2)]+A[(i+3)][k]*B[k][(j+2)];
                C[(i+4)][j]=C[(i+4)][j]+A[(i+4)][k]*B[k][j];
                C[(i+4)][(j+1)]=C[(i+4)][(j+1)]+A[(i+4)][k]*B[k][(j+1)];
                C[(i+4)][(j+2)]=C[(i+4)][(j+2)]+A[(i+4)][k]*B[k][(j+2)];
                C[(i+5)][j]=C[(i+5)][j]+A[(i+5)][k]*B[k][j];
                C[(i+5)][(j+1)]=C[(i+5)][(j+1)]+A[(i+5)][k]*B[k][(j+1)];
                C[(i+5)][(j+2)]=C[(i+5)][(j+2)]+A[(i+5)][k]*B[k][(j+2)];
                C[(i+6)][j]=C[(i+6)][j]+A[(i+6)][k]*B[k][j];
                C[(i+6)][(j+1)]=C[(i+6)][(j+1)]+A[(i+6)][k]*B[k][(j+1)];
                C[(i+6)][(j+2)]=C[(i+6)][(j+2)]+A[(i+6)][k]*B[k][(j+2)];
                C[(i+7)][j]=C[(i+7)][j]+A[(i+7)][k]*B[k][j];
                C[(i+7)][(j+1)]=C[(i+7)][(j+1)]+A[(i+7)][k]*B[k][(j+1)];
                C[(i+7)][(j+2)]=C[(i+7)][(j+2)]+A[(i+7)][k]*B[k][(j+2)];
                C[(i+8)][j]=C[(i+8)][j]+A[(i+8)][k]*B[k][j];
                C[(i+8)][(j+1)]=C[(i+8)][(j+1)]+A[(i+8)][k]*B[k][(j+1)];
                C[(i+8)][(j+2)]=C[(i+8)][(j+2)]+A[(i+8)][k]*B[k][(j+2)];
                C[(i+9)][j]=C[(i+9)][j]+A[(i+9)][k]*B[k][j];
                C[(i+9)][(j+1)]=C[(i+9)][(j+1)]+A[(i+9)][k]*B[k][(j+1)];
                C[(i+9)][(j+2)]=C[(i+9)][(j+2)]+A[(i+9)][k]*B[k][(j+2)];
                C[(i+10)][j]=C[(i+10)][j]+A[(i+10)][k]*B[k][j];
                C[(i+10)][(j+1)]=C[(i+10)][(j+1)]+A[(i+10)][k]*B[k][(j+1)];
                C[(i+10)][(j+2)]=C[(i+10)][(j+2)]+A[(i+10)][k]*B[k][(j+2)];
                C[(i+11)][j]=C[(i+11)][j]+A[(i+11)][k]*B[k][j];
                C[(i+11)][(j+1)]=C[(i+11)][(j+1)]+A[(i+11)][k]*B[k][(j+1)];
                C[(i+11)][(j+2)]=C[(i+11)][(j+2)]+A[(i+11)][k]*B[k][(j+2)];
                C[(i+12)][j]=C[(i+12)][j]+A[(i+12)][k]*B[k][j];
                C[(i+12)][(j+1)]=C[(i+12)][(j+1)]+A[(i+12)][k]*B[k][(j+1)];
                C[(i+12)][(j+2)]=C[(i+12)][(j+2)]+A[(i+12)][k]*B[k][(j+2)];
                C[(i+13)][j]=C[(i+13)][j]+A[(i+13)][k]*B[k][j];
                C[(i+13)][(j+1)]=C[(i+13)][(j+1)]+A[(i+13)][k]*B[k][(j+1)];
                C[(i+13)][(j+2)]=C[(i+13)][(j+2)]+A[(i+13)][k]*B[k][(j+2)];
                C[(i+14)][j]=C[(i+14)][j]+A[(i+14)][k]*B[k][j];
                C[(i+14)][(j+1)]=C[(i+14)][(j+1)]+A[(i+14)][k]*B[k][(j+1)];
                C[(i+14)][(j+2)]=C[(i+14)][(j+2)]+A[(i+14)][k]*B[k][(j+2)];
                C[(i+15)][j]=C[(i+15)][j]+A[(i+15)][k]*B[k][j];
                C[(i+15)][(j+1)]=C[(i+15)][(j+1)]+A[(i+15)][k]*B[k][(j+1)];
                C[(i+15)][(j+2)]=C[(i+15)][(j+2)]+A[(i+15)][k]*B[k][(j+2)];
                C[(i+16)][j]=C[(i+16)][j]+A[(i+16)][k]*B[k][j];
                C[(i+16)][(j+1)]=C[(i+16)][(j+1)]+A[(i+16)][k]*B[k][(j+1)];
                C[(i+16)][(j+2)]=C[(i+16)][(j+2)]+A[(i+16)][k]*B[k][(j+2)];
                C[(i+17)][j]=C[(i+17)][j]+A[(i+17)][k]*B[k][j];
                C[(i+17)][(j+1)]=C[(i+17)][(j+1)]+A[(i+17)][k]*B[k][(j+1)];
                C[(i+17)][(j+2)]=C[(i+17)][(j+2)]+A[(i+17)][k]*B[k][(j+2)];
                C[(i+18)][j]=C[(i+18)][j]+A[(i+18)][k]*B[k][j];
                C[(i+18)][(j+1)]=C[(i+18)][(j+1)]+A[(i+18)][k]*B[k][(j+1)];
                C[(i+18)][(j+2)]=C[(i+18)][(j+2)]+A[(i+18)][k]*B[k][(j+2)];
                C[(i+19)][j]=C[(i+19)][j]+A[(i+19)][k]*B[k][j];
                C[(i+19)][(j+1)]=C[(i+19)][(j+1)]+A[(i+19)][k]*B[k][(j+1)];
                C[(i+19)][(j+2)]=C[(i+19)][(j+2)]+A[(i+19)][k]*B[k][(j+2)];
                C[(i+20)][j]=C[(i+20)][j]+A[(i+20)][k]*B[k][j];
                C[(i+20)][(j+1)]=C[(i+20)][(j+1)]+A[(i+20)][k]*B[k][(j+1)];
                C[(i+20)][(j+2)]=C[(i+20)][(j+2)]+A[(i+20)][k]*B[k][(j+2)];
                C[(i+21)][j]=C[(i+21)][j]+A[(i+21)][k]*B[k][j];
                C[(i+21)][(j+1)]=C[(i+21)][(j+1)]+A[(i+21)][k]*B[k][(j+1)];
                C[(i+21)][(j+2)]=C[(i+21)][(j+2)]+A[(i+21)][k]*B[k][(j+2)];
                C[(i+22)][j]=C[(i+22)][j]+A[(i+22)][k]*B[k][j];
                C[(i+22)][(j+1)]=C[(i+22)][(j+1)]+A[(i+22)][k]*B[k][(j+1)];
                C[(i+22)][(j+2)]=C[(i+22)][(j+2)]+A[(i+22)][k]*B[k][(j+2)];
              }
              register int cbv_2;
              cbv_2=min(N-1,jj+15);
#pragma ivdep
#pragma vector always
              for (; j<=cbv_2; j=j+1) {
                C[i][j]=C[i][j]+A[i][k]*B[k][j];
                C[(i+1)][j]=C[(i+1)][j]+A[(i+1)][k]*B[k][j];
                C[(i+2)][j]=C[(i+2)][j]+A[(i+2)][k]*B[k][j];
                C[(i+3)][j]=C[(i+3)][j]+A[(i+3)][k]*B[k][j];
                C[(i+4)][j]=C[(i+4)][j]+A[(i+4)][k]*B[k][j];
                C[(i+5)][j]=C[(i+5)][j]+A[(i+5)][k]*B[k][j];
                C[(i+6)][j]=C[(i+6)][j]+A[(i+6)][k]*B[k][j];
                C[(i+7)][j]=C[(i+7)][j]+A[(i+7)][k]*B[k][j];
                C[(i+8)][j]=C[(i+8)][j]+A[(i+8)][k]*B[k][j];
                C[(i+9)][j]=C[(i+9)][j]+A[(i+9)][k]*B[k][j];
                C[(i+10)][j]=C[(i+10)][j]+A[(i+10)][k]*B[k][j];
                C[(i+11)][j]=C[(i+11)][j]+A[(i+11)][k]*B[k][j];
                C[(i+12)][j]=C[(i+12)][j]+A[(i+12)][k]*B[k][j];
                C[(i+13)][j]=C[(i+13)][j]+A[(i+13)][k]*B[k][j];
                C[(i+14)][j]=C[(i+14)][j]+A[(i+14)][k]*B[k][j];
                C[(i+15)][j]=C[(i+15)][j]+A[(i+15)][k]*B[k][j];
                C[(i+16)][j]=C[(i+16)][j]+A[(i+16)][k]*B[k][j];
                C[(i+17)][j]=C[(i+17)][j]+A[(i+17)][k]*B[k][j];
                C[(i+18)][j]=C[(i+18)][j]+A[(i+18)][k]*B[k][j];
                C[(i+19)][j]=C[(i+19)][j]+A[(i+19)][k]*B[k][j];
                C[(i+20)][j]=C[(i+20)][j]+A[(i+20)][k]*B[k][j];
                C[(i+21)][j]=C[(i+21)][j]+A[(i+21)][k]*B[k][j];
                C[(i+22)][j]=C[(i+22)][j]+A[(i+22)][k]*B[k][j];
              }
            }
          }
          for (; i<=min(M-1,iii+511); i=i+1) 
            for (k=kk; k<=min(K-1,kk+15); k=k+1) {
              register int cbv_3;
              cbv_3=min(N-1,jj+15)-2;
#pragma ivdep
#pragma vector always
              for (j=jj; j<=cbv_3; j=j+3) {
                C[i][j]=C[i][j]+A[i][k]*B[k][j];
                C[i][(j+1)]=C[i][(j+1)]+A[i][k]*B[k][(j+1)];
                C[i][(j+2)]=C[i][(j+2)]+A[i][k]*B[k][(j+2)];
              }
              register int cbv_4;
              cbv_4=min(N-1,jj+15);
#pragma ivdep
#pragma vector always
              for (; j<=cbv_4; j=j+1) 
                C[i][j]=C[i][j]+A[i][k]*B[k][j];
            }
        }
/*@ end @*/


    orio_t_end = getClock();
    orio_t = orio_t_end - orio_t_start;
    printf("{'[0, 1, 1, 4, 2, 3, 22, 2, 0, 0, 1, 0, 1, 19]' : %g}\n", orio_t);
    if (orio_t < orio_t_min) orio_t_min = orio_t;
  }
  
  /*
  printf("{'[0, 1, 1, 4, 2, 3, 22, 2, 0, 0, 1, 0, 1, 19]' : %g}", orio_t_min);
  */
  
  

  return 0;
}

