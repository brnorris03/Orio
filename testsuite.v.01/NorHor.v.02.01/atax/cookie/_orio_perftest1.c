
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#define N 10000
#include "decl.h"

#include "init.c"



extern double getClock(); 

int main(int argc, char *argv[])
{
  init_input_vars();


  double orio_t_start, orio_t_end, orio_t, orio_t_min = (double)LONG_MAX;
  double orio_times[ORIO_TIMES_ARRAY_SIZE];
  int orio_i;

  for (orio_i=0; orio_i<ORIO_REPS; orio_i++)
  {
    orio_t_start = getClock();
    
    

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

int i,j, k;
int it, jt, kt;
int ii, jj, kk;
int iii, jjj, kkk;

double* tmp=(double*) malloc(nx*sizeof(double));
  
/*@ begin Loop(

  transform Composite(
    unrolljam = (['i'],[U1_I]),
    vector = (VEC1, ['ivdep','vector always'])
  )
  for (i= 0; i<=ny-1; i++)
    y[i] = 0.0;

  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),('k',T1_K,'kk'),
            (('ii','i'),T2_I,'iii'),(('jj','j'),T2_J,'jjj'),(('kk','k'),T2_K,'kkk')],
    arrcopy = [(ACOPY_y, 'y[k]', [(T1_K if T1_K>1 else T2_K)],'_copy'),
               (ACOPY_x, 'x[j]', [(T1_J if T1_J>1 else T2_J)],'_copy')],
    unrolljam = (['k','j','i'],[U_K,U_J,U_I]),
    scalarreplace = (SCR, 'double'),
    regtile = (['i','j','k'],[RT_I,RT_J,RT_K]),
    vector = (VEC2, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k,y_copy,x_copy)')
  )
  for (i = 0; i<=nx-1; i++) {
    tmp[i] = 0;
    for (j = 0; j<=ny-1; j++) 
      tmp[i] = tmp[i] + A[i*ny+j]*x[j];
    for (k = 0; k<=ny-1; k++) 
      y[k] = y[k] + A[i*ny+k]*tmp[i];
  }
) @*/
{
  for (i=0; i<=ny-4; i=i+4) {
    y[i]=0.0;
    y[(i+1)]=0.0;
    y[(i+2)]=0.0;
    y[(i+3)]=0.0;
  }
  for (; i<=ny-1; i=i+1) 
    y[i]=0.0;
}
{
  double x_copy;
  double y_copy;
  for (iii=0; iii<=nx-1; iii=iii+256) 
    for (ii=iii; ii<=min(nx-1,iii); ii=ii+256) {
      for (it=ii; it<=min(nx-1,ii+255)-31; it=it+32) {
        tmp[it]=0;
        tmp[(it+1)]=0;
        tmp[(it+2)]=0;
        tmp[(it+3)]=0;
        tmp[(it+4)]=0;
        tmp[(it+5)]=0;
        tmp[(it+6)]=0;
        tmp[(it+7)]=0;
        tmp[(it+8)]=0;
        tmp[(it+9)]=0;
        tmp[(it+10)]=0;
        tmp[(it+11)]=0;
        tmp[(it+12)]=0;
        tmp[(it+13)]=0;
        tmp[(it+14)]=0;
        tmp[(it+15)]=0;
        tmp[(it+16)]=0;
        tmp[(it+17)]=0;
        tmp[(it+18)]=0;
        tmp[(it+19)]=0;
        tmp[(it+20)]=0;
        tmp[(it+21)]=0;
        tmp[(it+22)]=0;
        tmp[(it+23)]=0;
        tmp[(it+24)]=0;
        tmp[(it+25)]=0;
        tmp[(it+26)]=0;
        tmp[(it+27)]=0;
        tmp[(it+28)]=0;
        tmp[(it+29)]=0;
        tmp[(it+30)]=0;
        tmp[(it+31)]=0;
        for (jjj=0; jjj<=ny-1; jjj=jjj+1024) 
          for (jj=jjj; jj<=min(ny-1,jjj+896); jj=jj+128) {
            register int cbv_1;
            cbv_1=min(ny-1,jj+127);
#pragma ivdep
#pragma vector always
            for (j=jj; j<=cbv_1; j=j+1) {
              tmp[it]=tmp[it]+A[it*ny+j]*x[j];
              tmp[(it+1)]=tmp[(it+1)]+A[(it+1)*ny+j]*x[j];
              tmp[(it+2)]=tmp[(it+2)]+A[(it+2)*ny+j]*x[j];
              tmp[(it+3)]=tmp[(it+3)]+A[(it+3)*ny+j]*x[j];
              tmp[(it+4)]=tmp[(it+4)]+A[(it+4)*ny+j]*x[j];
              tmp[(it+5)]=tmp[(it+5)]+A[(it+5)*ny+j]*x[j];
              tmp[(it+6)]=tmp[(it+6)]+A[(it+6)*ny+j]*x[j];
              tmp[(it+7)]=tmp[(it+7)]+A[(it+7)*ny+j]*x[j];
              tmp[(it+8)]=tmp[(it+8)]+A[(it+8)*ny+j]*x[j];
              tmp[(it+9)]=tmp[(it+9)]+A[(it+9)*ny+j]*x[j];
              tmp[(it+10)]=tmp[(it+10)]+A[(it+10)*ny+j]*x[j];
              tmp[(it+11)]=tmp[(it+11)]+A[(it+11)*ny+j]*x[j];
              tmp[(it+12)]=tmp[(it+12)]+A[(it+12)*ny+j]*x[j];
              tmp[(it+13)]=tmp[(it+13)]+A[(it+13)*ny+j]*x[j];
              tmp[(it+14)]=tmp[(it+14)]+A[(it+14)*ny+j]*x[j];
              tmp[(it+15)]=tmp[(it+15)]+A[(it+15)*ny+j]*x[j];
              tmp[(it+16)]=tmp[(it+16)]+A[(it+16)*ny+j]*x[j];
              tmp[(it+17)]=tmp[(it+17)]+A[(it+17)*ny+j]*x[j];
              tmp[(it+18)]=tmp[(it+18)]+A[(it+18)*ny+j]*x[j];
              tmp[(it+19)]=tmp[(it+19)]+A[(it+19)*ny+j]*x[j];
              tmp[(it+20)]=tmp[(it+20)]+A[(it+20)*ny+j]*x[j];
              tmp[(it+21)]=tmp[(it+21)]+A[(it+21)*ny+j]*x[j];
              tmp[(it+22)]=tmp[(it+22)]+A[(it+22)*ny+j]*x[j];
              tmp[(it+23)]=tmp[(it+23)]+A[(it+23)*ny+j]*x[j];
              tmp[(it+24)]=tmp[(it+24)]+A[(it+24)*ny+j]*x[j];
              tmp[(it+25)]=tmp[(it+25)]+A[(it+25)*ny+j]*x[j];
              tmp[(it+26)]=tmp[(it+26)]+A[(it+26)*ny+j]*x[j];
              tmp[(it+27)]=tmp[(it+27)]+A[(it+27)*ny+j]*x[j];
              tmp[(it+28)]=tmp[(it+28)]+A[(it+28)*ny+j]*x[j];
              tmp[(it+29)]=tmp[(it+29)]+A[(it+29)*ny+j]*x[j];
              tmp[(it+30)]=tmp[(it+30)]+A[(it+30)*ny+j]*x[j];
              tmp[(it+31)]=tmp[(it+31)]+A[(it+31)*ny+j]*x[j];
            }
          }
        for (kkk=0; kkk<=ny-1; kkk=kkk+64) 
          for (kk=kkk; kk<=min(ny-1,kkk+32); kk=kk+32) {
            register int cbv_2;
            cbv_2=min(ny-1,kk+31)-7;
#pragma ivdep
#pragma vector always
            for (k=kk; k<=cbv_2; k=k+8) {
              y[k]=y[k]+A[it*ny+k]*tmp[it];
              y[k]=y[k]+A[(it+1)*ny+k]*tmp[(it+1)];
              y[k]=y[k]+A[(it+2)*ny+k]*tmp[(it+2)];
              y[k]=y[k]+A[(it+3)*ny+k]*tmp[(it+3)];
              y[k]=y[k]+A[(it+4)*ny+k]*tmp[(it+4)];
              y[k]=y[k]+A[(it+5)*ny+k]*tmp[(it+5)];
              y[k]=y[k]+A[(it+6)*ny+k]*tmp[(it+6)];
              y[k]=y[k]+A[(it+7)*ny+k]*tmp[(it+7)];
              y[k]=y[k]+A[(it+8)*ny+k]*tmp[(it+8)];
              y[k]=y[k]+A[(it+9)*ny+k]*tmp[(it+9)];
              y[k]=y[k]+A[(it+10)*ny+k]*tmp[(it+10)];
              y[k]=y[k]+A[(it+11)*ny+k]*tmp[(it+11)];
              y[k]=y[k]+A[(it+12)*ny+k]*tmp[(it+12)];
              y[k]=y[k]+A[(it+13)*ny+k]*tmp[(it+13)];
              y[k]=y[k]+A[(it+14)*ny+k]*tmp[(it+14)];
              y[k]=y[k]+A[(it+15)*ny+k]*tmp[(it+15)];
              y[k]=y[k]+A[(it+16)*ny+k]*tmp[(it+16)];
              y[k]=y[k]+A[(it+17)*ny+k]*tmp[(it+17)];
              y[k]=y[k]+A[(it+18)*ny+k]*tmp[(it+18)];
              y[k]=y[k]+A[(it+19)*ny+k]*tmp[(it+19)];
              y[k]=y[k]+A[(it+20)*ny+k]*tmp[(it+20)];
              y[k]=y[k]+A[(it+21)*ny+k]*tmp[(it+21)];
              y[k]=y[k]+A[(it+22)*ny+k]*tmp[(it+22)];
              y[k]=y[k]+A[(it+23)*ny+k]*tmp[(it+23)];
              y[k]=y[k]+A[(it+24)*ny+k]*tmp[(it+24)];
              y[k]=y[k]+A[(it+25)*ny+k]*tmp[(it+25)];
              y[k]=y[k]+A[(it+26)*ny+k]*tmp[(it+26)];
              y[k]=y[k]+A[(it+27)*ny+k]*tmp[(it+27)];
              y[k]=y[k]+A[(it+28)*ny+k]*tmp[(it+28)];
              y[k]=y[k]+A[(it+29)*ny+k]*tmp[(it+29)];
              y[k]=y[k]+A[(it+30)*ny+k]*tmp[(it+30)];
              y[k]=y[k]+A[(it+31)*ny+k]*tmp[(it+31)];
              y[(k+1)]=y[(k+1)]+A[it*ny+k+1]*tmp[it];
              y[(k+1)]=y[(k+1)]+A[(it+1)*ny+k+1]*tmp[(it+1)];
              y[(k+1)]=y[(k+1)]+A[(it+2)*ny+k+1]*tmp[(it+2)];
              y[(k+1)]=y[(k+1)]+A[(it+3)*ny+k+1]*tmp[(it+3)];
              y[(k+1)]=y[(k+1)]+A[(it+4)*ny+k+1]*tmp[(it+4)];
              y[(k+1)]=y[(k+1)]+A[(it+5)*ny+k+1]*tmp[(it+5)];
              y[(k+1)]=y[(k+1)]+A[(it+6)*ny+k+1]*tmp[(it+6)];
              y[(k+1)]=y[(k+1)]+A[(it+7)*ny+k+1]*tmp[(it+7)];
              y[(k+1)]=y[(k+1)]+A[(it+8)*ny+k+1]*tmp[(it+8)];
              y[(k+1)]=y[(k+1)]+A[(it+9)*ny+k+1]*tmp[(it+9)];
              y[(k+1)]=y[(k+1)]+A[(it+10)*ny+k+1]*tmp[(it+10)];
              y[(k+1)]=y[(k+1)]+A[(it+11)*ny+k+1]*tmp[(it+11)];
              y[(k+1)]=y[(k+1)]+A[(it+12)*ny+k+1]*tmp[(it+12)];
              y[(k+1)]=y[(k+1)]+A[(it+13)*ny+k+1]*tmp[(it+13)];
              y[(k+1)]=y[(k+1)]+A[(it+14)*ny+k+1]*tmp[(it+14)];
              y[(k+1)]=y[(k+1)]+A[(it+15)*ny+k+1]*tmp[(it+15)];
              y[(k+1)]=y[(k+1)]+A[(it+16)*ny+k+1]*tmp[(it+16)];
              y[(k+1)]=y[(k+1)]+A[(it+17)*ny+k+1]*tmp[(it+17)];
              y[(k+1)]=y[(k+1)]+A[(it+18)*ny+k+1]*tmp[(it+18)];
              y[(k+1)]=y[(k+1)]+A[(it+19)*ny+k+1]*tmp[(it+19)];
              y[(k+1)]=y[(k+1)]+A[(it+20)*ny+k+1]*tmp[(it+20)];
              y[(k+1)]=y[(k+1)]+A[(it+21)*ny+k+1]*tmp[(it+21)];
              y[(k+1)]=y[(k+1)]+A[(it+22)*ny+k+1]*tmp[(it+22)];
              y[(k+1)]=y[(k+1)]+A[(it+23)*ny+k+1]*tmp[(it+23)];
              y[(k+1)]=y[(k+1)]+A[(it+24)*ny+k+1]*tmp[(it+24)];
              y[(k+1)]=y[(k+1)]+A[(it+25)*ny+k+1]*tmp[(it+25)];
              y[(k+1)]=y[(k+1)]+A[(it+26)*ny+k+1]*tmp[(it+26)];
              y[(k+1)]=y[(k+1)]+A[(it+27)*ny+k+1]*tmp[(it+27)];
              y[(k+1)]=y[(k+1)]+A[(it+28)*ny+k+1]*tmp[(it+28)];
              y[(k+1)]=y[(k+1)]+A[(it+29)*ny+k+1]*tmp[(it+29)];
              y[(k+1)]=y[(k+1)]+A[(it+30)*ny+k+1]*tmp[(it+30)];
              y[(k+1)]=y[(k+1)]+A[(it+31)*ny+k+1]*tmp[(it+31)];
              y[(k+2)]=y[(k+2)]+A[it*ny+k+2]*tmp[it];
              y[(k+2)]=y[(k+2)]+A[(it+1)*ny+k+2]*tmp[(it+1)];
              y[(k+2)]=y[(k+2)]+A[(it+2)*ny+k+2]*tmp[(it+2)];
              y[(k+2)]=y[(k+2)]+A[(it+3)*ny+k+2]*tmp[(it+3)];
              y[(k+2)]=y[(k+2)]+A[(it+4)*ny+k+2]*tmp[(it+4)];
              y[(k+2)]=y[(k+2)]+A[(it+5)*ny+k+2]*tmp[(it+5)];
              y[(k+2)]=y[(k+2)]+A[(it+6)*ny+k+2]*tmp[(it+6)];
              y[(k+2)]=y[(k+2)]+A[(it+7)*ny+k+2]*tmp[(it+7)];
              y[(k+2)]=y[(k+2)]+A[(it+8)*ny+k+2]*tmp[(it+8)];
              y[(k+2)]=y[(k+2)]+A[(it+9)*ny+k+2]*tmp[(it+9)];
              y[(k+2)]=y[(k+2)]+A[(it+10)*ny+k+2]*tmp[(it+10)];
              y[(k+2)]=y[(k+2)]+A[(it+11)*ny+k+2]*tmp[(it+11)];
              y[(k+2)]=y[(k+2)]+A[(it+12)*ny+k+2]*tmp[(it+12)];
              y[(k+2)]=y[(k+2)]+A[(it+13)*ny+k+2]*tmp[(it+13)];
              y[(k+2)]=y[(k+2)]+A[(it+14)*ny+k+2]*tmp[(it+14)];
              y[(k+2)]=y[(k+2)]+A[(it+15)*ny+k+2]*tmp[(it+15)];
              y[(k+2)]=y[(k+2)]+A[(it+16)*ny+k+2]*tmp[(it+16)];
              y[(k+2)]=y[(k+2)]+A[(it+17)*ny+k+2]*tmp[(it+17)];
              y[(k+2)]=y[(k+2)]+A[(it+18)*ny+k+2]*tmp[(it+18)];
              y[(k+2)]=y[(k+2)]+A[(it+19)*ny+k+2]*tmp[(it+19)];
              y[(k+2)]=y[(k+2)]+A[(it+20)*ny+k+2]*tmp[(it+20)];
              y[(k+2)]=y[(k+2)]+A[(it+21)*ny+k+2]*tmp[(it+21)];
              y[(k+2)]=y[(k+2)]+A[(it+22)*ny+k+2]*tmp[(it+22)];
              y[(k+2)]=y[(k+2)]+A[(it+23)*ny+k+2]*tmp[(it+23)];
              y[(k+2)]=y[(k+2)]+A[(it+24)*ny+k+2]*tmp[(it+24)];
              y[(k+2)]=y[(k+2)]+A[(it+25)*ny+k+2]*tmp[(it+25)];
              y[(k+2)]=y[(k+2)]+A[(it+26)*ny+k+2]*tmp[(it+26)];
              y[(k+2)]=y[(k+2)]+A[(it+27)*ny+k+2]*tmp[(it+27)];
              y[(k+2)]=y[(k+2)]+A[(it+28)*ny+k+2]*tmp[(it+28)];
              y[(k+2)]=y[(k+2)]+A[(it+29)*ny+k+2]*tmp[(it+29)];
              y[(k+2)]=y[(k+2)]+A[(it+30)*ny+k+2]*tmp[(it+30)];
              y[(k+2)]=y[(k+2)]+A[(it+31)*ny+k+2]*tmp[(it+31)];
              y[(k+3)]=y[(k+3)]+A[it*ny+k+3]*tmp[it];
              y[(k+3)]=y[(k+3)]+A[(it+1)*ny+k+3]*tmp[(it+1)];
              y[(k+3)]=y[(k+3)]+A[(it+2)*ny+k+3]*tmp[(it+2)];
              y[(k+3)]=y[(k+3)]+A[(it+3)*ny+k+3]*tmp[(it+3)];
              y[(k+3)]=y[(k+3)]+A[(it+4)*ny+k+3]*tmp[(it+4)];
              y[(k+3)]=y[(k+3)]+A[(it+5)*ny+k+3]*tmp[(it+5)];
              y[(k+3)]=y[(k+3)]+A[(it+6)*ny+k+3]*tmp[(it+6)];
              y[(k+3)]=y[(k+3)]+A[(it+7)*ny+k+3]*tmp[(it+7)];
              y[(k+3)]=y[(k+3)]+A[(it+8)*ny+k+3]*tmp[(it+8)];
              y[(k+3)]=y[(k+3)]+A[(it+9)*ny+k+3]*tmp[(it+9)];
              y[(k+3)]=y[(k+3)]+A[(it+10)*ny+k+3]*tmp[(it+10)];
              y[(k+3)]=y[(k+3)]+A[(it+11)*ny+k+3]*tmp[(it+11)];
              y[(k+3)]=y[(k+3)]+A[(it+12)*ny+k+3]*tmp[(it+12)];
              y[(k+3)]=y[(k+3)]+A[(it+13)*ny+k+3]*tmp[(it+13)];
              y[(k+3)]=y[(k+3)]+A[(it+14)*ny+k+3]*tmp[(it+14)];
              y[(k+3)]=y[(k+3)]+A[(it+15)*ny+k+3]*tmp[(it+15)];
              y[(k+3)]=y[(k+3)]+A[(it+16)*ny+k+3]*tmp[(it+16)];
              y[(k+3)]=y[(k+3)]+A[(it+17)*ny+k+3]*tmp[(it+17)];
              y[(k+3)]=y[(k+3)]+A[(it+18)*ny+k+3]*tmp[(it+18)];
              y[(k+3)]=y[(k+3)]+A[(it+19)*ny+k+3]*tmp[(it+19)];
              y[(k+3)]=y[(k+3)]+A[(it+20)*ny+k+3]*tmp[(it+20)];
              y[(k+3)]=y[(k+3)]+A[(it+21)*ny+k+3]*tmp[(it+21)];
              y[(k+3)]=y[(k+3)]+A[(it+22)*ny+k+3]*tmp[(it+22)];
              y[(k+3)]=y[(k+3)]+A[(it+23)*ny+k+3]*tmp[(it+23)];
              y[(k+3)]=y[(k+3)]+A[(it+24)*ny+k+3]*tmp[(it+24)];
              y[(k+3)]=y[(k+3)]+A[(it+25)*ny+k+3]*tmp[(it+25)];
              y[(k+3)]=y[(k+3)]+A[(it+26)*ny+k+3]*tmp[(it+26)];
              y[(k+3)]=y[(k+3)]+A[(it+27)*ny+k+3]*tmp[(it+27)];
              y[(k+3)]=y[(k+3)]+A[(it+28)*ny+k+3]*tmp[(it+28)];
              y[(k+3)]=y[(k+3)]+A[(it+29)*ny+k+3]*tmp[(it+29)];
              y[(k+3)]=y[(k+3)]+A[(it+30)*ny+k+3]*tmp[(it+30)];
              y[(k+3)]=y[(k+3)]+A[(it+31)*ny+k+3]*tmp[(it+31)];
              y[(k+4)]=y[(k+4)]+A[it*ny+k+4]*tmp[it];
              y[(k+4)]=y[(k+4)]+A[(it+1)*ny+k+4]*tmp[(it+1)];
              y[(k+4)]=y[(k+4)]+A[(it+2)*ny+k+4]*tmp[(it+2)];
              y[(k+4)]=y[(k+4)]+A[(it+3)*ny+k+4]*tmp[(it+3)];
              y[(k+4)]=y[(k+4)]+A[(it+4)*ny+k+4]*tmp[(it+4)];
              y[(k+4)]=y[(k+4)]+A[(it+5)*ny+k+4]*tmp[(it+5)];
              y[(k+4)]=y[(k+4)]+A[(it+6)*ny+k+4]*tmp[(it+6)];
              y[(k+4)]=y[(k+4)]+A[(it+7)*ny+k+4]*tmp[(it+7)];
              y[(k+4)]=y[(k+4)]+A[(it+8)*ny+k+4]*tmp[(it+8)];
              y[(k+4)]=y[(k+4)]+A[(it+9)*ny+k+4]*tmp[(it+9)];
              y[(k+4)]=y[(k+4)]+A[(it+10)*ny+k+4]*tmp[(it+10)];
              y[(k+4)]=y[(k+4)]+A[(it+11)*ny+k+4]*tmp[(it+11)];
              y[(k+4)]=y[(k+4)]+A[(it+12)*ny+k+4]*tmp[(it+12)];
              y[(k+4)]=y[(k+4)]+A[(it+13)*ny+k+4]*tmp[(it+13)];
              y[(k+4)]=y[(k+4)]+A[(it+14)*ny+k+4]*tmp[(it+14)];
              y[(k+4)]=y[(k+4)]+A[(it+15)*ny+k+4]*tmp[(it+15)];
              y[(k+4)]=y[(k+4)]+A[(it+16)*ny+k+4]*tmp[(it+16)];
              y[(k+4)]=y[(k+4)]+A[(it+17)*ny+k+4]*tmp[(it+17)];
              y[(k+4)]=y[(k+4)]+A[(it+18)*ny+k+4]*tmp[(it+18)];
              y[(k+4)]=y[(k+4)]+A[(it+19)*ny+k+4]*tmp[(it+19)];
              y[(k+4)]=y[(k+4)]+A[(it+20)*ny+k+4]*tmp[(it+20)];
              y[(k+4)]=y[(k+4)]+A[(it+21)*ny+k+4]*tmp[(it+21)];
              y[(k+4)]=y[(k+4)]+A[(it+22)*ny+k+4]*tmp[(it+22)];
              y[(k+4)]=y[(k+4)]+A[(it+23)*ny+k+4]*tmp[(it+23)];
              y[(k+4)]=y[(k+4)]+A[(it+24)*ny+k+4]*tmp[(it+24)];
              y[(k+4)]=y[(k+4)]+A[(it+25)*ny+k+4]*tmp[(it+25)];
              y[(k+4)]=y[(k+4)]+A[(it+26)*ny+k+4]*tmp[(it+26)];
              y[(k+4)]=y[(k+4)]+A[(it+27)*ny+k+4]*tmp[(it+27)];
              y[(k+4)]=y[(k+4)]+A[(it+28)*ny+k+4]*tmp[(it+28)];
              y[(k+4)]=y[(k+4)]+A[(it+29)*ny+k+4]*tmp[(it+29)];
              y[(k+4)]=y[(k+4)]+A[(it+30)*ny+k+4]*tmp[(it+30)];
              y[(k+4)]=y[(k+4)]+A[(it+31)*ny+k+4]*tmp[(it+31)];
              y[(k+5)]=y[(k+5)]+A[it*ny+k+5]*tmp[it];
              y[(k+5)]=y[(k+5)]+A[(it+1)*ny+k+5]*tmp[(it+1)];
              y[(k+5)]=y[(k+5)]+A[(it+2)*ny+k+5]*tmp[(it+2)];
              y[(k+5)]=y[(k+5)]+A[(it+3)*ny+k+5]*tmp[(it+3)];
              y[(k+5)]=y[(k+5)]+A[(it+4)*ny+k+5]*tmp[(it+4)];
              y[(k+5)]=y[(k+5)]+A[(it+5)*ny+k+5]*tmp[(it+5)];
              y[(k+5)]=y[(k+5)]+A[(it+6)*ny+k+5]*tmp[(it+6)];
              y[(k+5)]=y[(k+5)]+A[(it+7)*ny+k+5]*tmp[(it+7)];
              y[(k+5)]=y[(k+5)]+A[(it+8)*ny+k+5]*tmp[(it+8)];
              y[(k+5)]=y[(k+5)]+A[(it+9)*ny+k+5]*tmp[(it+9)];
              y[(k+5)]=y[(k+5)]+A[(it+10)*ny+k+5]*tmp[(it+10)];
              y[(k+5)]=y[(k+5)]+A[(it+11)*ny+k+5]*tmp[(it+11)];
              y[(k+5)]=y[(k+5)]+A[(it+12)*ny+k+5]*tmp[(it+12)];
              y[(k+5)]=y[(k+5)]+A[(it+13)*ny+k+5]*tmp[(it+13)];
              y[(k+5)]=y[(k+5)]+A[(it+14)*ny+k+5]*tmp[(it+14)];
              y[(k+5)]=y[(k+5)]+A[(it+15)*ny+k+5]*tmp[(it+15)];
              y[(k+5)]=y[(k+5)]+A[(it+16)*ny+k+5]*tmp[(it+16)];
              y[(k+5)]=y[(k+5)]+A[(it+17)*ny+k+5]*tmp[(it+17)];
              y[(k+5)]=y[(k+5)]+A[(it+18)*ny+k+5]*tmp[(it+18)];
              y[(k+5)]=y[(k+5)]+A[(it+19)*ny+k+5]*tmp[(it+19)];
              y[(k+5)]=y[(k+5)]+A[(it+20)*ny+k+5]*tmp[(it+20)];
              y[(k+5)]=y[(k+5)]+A[(it+21)*ny+k+5]*tmp[(it+21)];
              y[(k+5)]=y[(k+5)]+A[(it+22)*ny+k+5]*tmp[(it+22)];
              y[(k+5)]=y[(k+5)]+A[(it+23)*ny+k+5]*tmp[(it+23)];
              y[(k+5)]=y[(k+5)]+A[(it+24)*ny+k+5]*tmp[(it+24)];
              y[(k+5)]=y[(k+5)]+A[(it+25)*ny+k+5]*tmp[(it+25)];
              y[(k+5)]=y[(k+5)]+A[(it+26)*ny+k+5]*tmp[(it+26)];
              y[(k+5)]=y[(k+5)]+A[(it+27)*ny+k+5]*tmp[(it+27)];
              y[(k+5)]=y[(k+5)]+A[(it+28)*ny+k+5]*tmp[(it+28)];
              y[(k+5)]=y[(k+5)]+A[(it+29)*ny+k+5]*tmp[(it+29)];
              y[(k+5)]=y[(k+5)]+A[(it+30)*ny+k+5]*tmp[(it+30)];
              y[(k+5)]=y[(k+5)]+A[(it+31)*ny+k+5]*tmp[(it+31)];
              y[(k+6)]=y[(k+6)]+A[it*ny+k+6]*tmp[it];
              y[(k+6)]=y[(k+6)]+A[(it+1)*ny+k+6]*tmp[(it+1)];
              y[(k+6)]=y[(k+6)]+A[(it+2)*ny+k+6]*tmp[(it+2)];
              y[(k+6)]=y[(k+6)]+A[(it+3)*ny+k+6]*tmp[(it+3)];
              y[(k+6)]=y[(k+6)]+A[(it+4)*ny+k+6]*tmp[(it+4)];
              y[(k+6)]=y[(k+6)]+A[(it+5)*ny+k+6]*tmp[(it+5)];
              y[(k+6)]=y[(k+6)]+A[(it+6)*ny+k+6]*tmp[(it+6)];
              y[(k+6)]=y[(k+6)]+A[(it+7)*ny+k+6]*tmp[(it+7)];
              y[(k+6)]=y[(k+6)]+A[(it+8)*ny+k+6]*tmp[(it+8)];
              y[(k+6)]=y[(k+6)]+A[(it+9)*ny+k+6]*tmp[(it+9)];
              y[(k+6)]=y[(k+6)]+A[(it+10)*ny+k+6]*tmp[(it+10)];
              y[(k+6)]=y[(k+6)]+A[(it+11)*ny+k+6]*tmp[(it+11)];
              y[(k+6)]=y[(k+6)]+A[(it+12)*ny+k+6]*tmp[(it+12)];
              y[(k+6)]=y[(k+6)]+A[(it+13)*ny+k+6]*tmp[(it+13)];
              y[(k+6)]=y[(k+6)]+A[(it+14)*ny+k+6]*tmp[(it+14)];
              y[(k+6)]=y[(k+6)]+A[(it+15)*ny+k+6]*tmp[(it+15)];
              y[(k+6)]=y[(k+6)]+A[(it+16)*ny+k+6]*tmp[(it+16)];
              y[(k+6)]=y[(k+6)]+A[(it+17)*ny+k+6]*tmp[(it+17)];
              y[(k+6)]=y[(k+6)]+A[(it+18)*ny+k+6]*tmp[(it+18)];
              y[(k+6)]=y[(k+6)]+A[(it+19)*ny+k+6]*tmp[(it+19)];
              y[(k+6)]=y[(k+6)]+A[(it+20)*ny+k+6]*tmp[(it+20)];
              y[(k+6)]=y[(k+6)]+A[(it+21)*ny+k+6]*tmp[(it+21)];
              y[(k+6)]=y[(k+6)]+A[(it+22)*ny+k+6]*tmp[(it+22)];
              y[(k+6)]=y[(k+6)]+A[(it+23)*ny+k+6]*tmp[(it+23)];
              y[(k+6)]=y[(k+6)]+A[(it+24)*ny+k+6]*tmp[(it+24)];
              y[(k+6)]=y[(k+6)]+A[(it+25)*ny+k+6]*tmp[(it+25)];
              y[(k+6)]=y[(k+6)]+A[(it+26)*ny+k+6]*tmp[(it+26)];
              y[(k+6)]=y[(k+6)]+A[(it+27)*ny+k+6]*tmp[(it+27)];
              y[(k+6)]=y[(k+6)]+A[(it+28)*ny+k+6]*tmp[(it+28)];
              y[(k+6)]=y[(k+6)]+A[(it+29)*ny+k+6]*tmp[(it+29)];
              y[(k+6)]=y[(k+6)]+A[(it+30)*ny+k+6]*tmp[(it+30)];
              y[(k+6)]=y[(k+6)]+A[(it+31)*ny+k+6]*tmp[(it+31)];
              y[(k+7)]=y[(k+7)]+A[it*ny+k+7]*tmp[it];
              y[(k+7)]=y[(k+7)]+A[(it+1)*ny+k+7]*tmp[(it+1)];
              y[(k+7)]=y[(k+7)]+A[(it+2)*ny+k+7]*tmp[(it+2)];
              y[(k+7)]=y[(k+7)]+A[(it+3)*ny+k+7]*tmp[(it+3)];
              y[(k+7)]=y[(k+7)]+A[(it+4)*ny+k+7]*tmp[(it+4)];
              y[(k+7)]=y[(k+7)]+A[(it+5)*ny+k+7]*tmp[(it+5)];
              y[(k+7)]=y[(k+7)]+A[(it+6)*ny+k+7]*tmp[(it+6)];
              y[(k+7)]=y[(k+7)]+A[(it+7)*ny+k+7]*tmp[(it+7)];
              y[(k+7)]=y[(k+7)]+A[(it+8)*ny+k+7]*tmp[(it+8)];
              y[(k+7)]=y[(k+7)]+A[(it+9)*ny+k+7]*tmp[(it+9)];
              y[(k+7)]=y[(k+7)]+A[(it+10)*ny+k+7]*tmp[(it+10)];
              y[(k+7)]=y[(k+7)]+A[(it+11)*ny+k+7]*tmp[(it+11)];
              y[(k+7)]=y[(k+7)]+A[(it+12)*ny+k+7]*tmp[(it+12)];
              y[(k+7)]=y[(k+7)]+A[(it+13)*ny+k+7]*tmp[(it+13)];
              y[(k+7)]=y[(k+7)]+A[(it+14)*ny+k+7]*tmp[(it+14)];
              y[(k+7)]=y[(k+7)]+A[(it+15)*ny+k+7]*tmp[(it+15)];
              y[(k+7)]=y[(k+7)]+A[(it+16)*ny+k+7]*tmp[(it+16)];
              y[(k+7)]=y[(k+7)]+A[(it+17)*ny+k+7]*tmp[(it+17)];
              y[(k+7)]=y[(k+7)]+A[(it+18)*ny+k+7]*tmp[(it+18)];
              y[(k+7)]=y[(k+7)]+A[(it+19)*ny+k+7]*tmp[(it+19)];
              y[(k+7)]=y[(k+7)]+A[(it+20)*ny+k+7]*tmp[(it+20)];
              y[(k+7)]=y[(k+7)]+A[(it+21)*ny+k+7]*tmp[(it+21)];
              y[(k+7)]=y[(k+7)]+A[(it+22)*ny+k+7]*tmp[(it+22)];
              y[(k+7)]=y[(k+7)]+A[(it+23)*ny+k+7]*tmp[(it+23)];
              y[(k+7)]=y[(k+7)]+A[(it+24)*ny+k+7]*tmp[(it+24)];
              y[(k+7)]=y[(k+7)]+A[(it+25)*ny+k+7]*tmp[(it+25)];
              y[(k+7)]=y[(k+7)]+A[(it+26)*ny+k+7]*tmp[(it+26)];
              y[(k+7)]=y[(k+7)]+A[(it+27)*ny+k+7]*tmp[(it+27)];
              y[(k+7)]=y[(k+7)]+A[(it+28)*ny+k+7]*tmp[(it+28)];
              y[(k+7)]=y[(k+7)]+A[(it+29)*ny+k+7]*tmp[(it+29)];
              y[(k+7)]=y[(k+7)]+A[(it+30)*ny+k+7]*tmp[(it+30)];
              y[(k+7)]=y[(k+7)]+A[(it+31)*ny+k+7]*tmp[(it+31)];
            }
            register int cbv_3;
            cbv_3=min(ny-1,kk+31);
#pragma ivdep
#pragma vector always
            for (; k<=cbv_3; k=k+1) {
              y[k]=y[k]+A[it*ny+k]*tmp[it];
              y[k]=y[k]+A[(it+1)*ny+k]*tmp[(it+1)];
              y[k]=y[k]+A[(it+2)*ny+k]*tmp[(it+2)];
              y[k]=y[k]+A[(it+3)*ny+k]*tmp[(it+3)];
              y[k]=y[k]+A[(it+4)*ny+k]*tmp[(it+4)];
              y[k]=y[k]+A[(it+5)*ny+k]*tmp[(it+5)];
              y[k]=y[k]+A[(it+6)*ny+k]*tmp[(it+6)];
              y[k]=y[k]+A[(it+7)*ny+k]*tmp[(it+7)];
              y[k]=y[k]+A[(it+8)*ny+k]*tmp[(it+8)];
              y[k]=y[k]+A[(it+9)*ny+k]*tmp[(it+9)];
              y[k]=y[k]+A[(it+10)*ny+k]*tmp[(it+10)];
              y[k]=y[k]+A[(it+11)*ny+k]*tmp[(it+11)];
              y[k]=y[k]+A[(it+12)*ny+k]*tmp[(it+12)];
              y[k]=y[k]+A[(it+13)*ny+k]*tmp[(it+13)];
              y[k]=y[k]+A[(it+14)*ny+k]*tmp[(it+14)];
              y[k]=y[k]+A[(it+15)*ny+k]*tmp[(it+15)];
              y[k]=y[k]+A[(it+16)*ny+k]*tmp[(it+16)];
              y[k]=y[k]+A[(it+17)*ny+k]*tmp[(it+17)];
              y[k]=y[k]+A[(it+18)*ny+k]*tmp[(it+18)];
              y[k]=y[k]+A[(it+19)*ny+k]*tmp[(it+19)];
              y[k]=y[k]+A[(it+20)*ny+k]*tmp[(it+20)];
              y[k]=y[k]+A[(it+21)*ny+k]*tmp[(it+21)];
              y[k]=y[k]+A[(it+22)*ny+k]*tmp[(it+22)];
              y[k]=y[k]+A[(it+23)*ny+k]*tmp[(it+23)];
              y[k]=y[k]+A[(it+24)*ny+k]*tmp[(it+24)];
              y[k]=y[k]+A[(it+25)*ny+k]*tmp[(it+25)];
              y[k]=y[k]+A[(it+26)*ny+k]*tmp[(it+26)];
              y[k]=y[k]+A[(it+27)*ny+k]*tmp[(it+27)];
              y[k]=y[k]+A[(it+28)*ny+k]*tmp[(it+28)];
              y[k]=y[k]+A[(it+29)*ny+k]*tmp[(it+29)];
              y[k]=y[k]+A[(it+30)*ny+k]*tmp[(it+30)];
              y[k]=y[k]+A[(it+31)*ny+k]*tmp[(it+31)];
            }
          }
      }
      {
        for (i=it; i<=min(nx-1,ii+255)-7; i=i+8) {
          tmp[i]=0;
          tmp[(i+1)]=0;
          tmp[(i+2)]=0;
          tmp[(i+3)]=0;
          tmp[(i+4)]=0;
          tmp[(i+5)]=0;
          tmp[(i+6)]=0;
          tmp[(i+7)]=0;
          for (jjj=0; jjj<=ny-1; jjj=jjj+1024) {
            for (jj=jjj; jj<=min(ny-1,jjj+896); jj=jj+128) {
              register int cbv_4;
              cbv_4=min(ny-1,jj+127);
#pragma ivdep
#pragma vector always
              for (j=jj; j<=cbv_4; j=j+1) {
                tmp[i]=tmp[i]+A[i*ny+j]*x[j];
                tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j]*x[j];
                tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j]*x[j];
                tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j]*x[j];
                tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j]*x[j];
                tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j]*x[j];
                tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j]*x[j];
                tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j]*x[j];
              }
            }
          }
          for (kkk=0; kkk<=ny-1; kkk=kkk+64) {
            for (kk=kkk; kk<=min(ny-1,kkk+32); kk=kk+32) {
              register int cbv_5;
              cbv_5=min(ny-1,kk+31)-7;
#pragma ivdep
#pragma vector always
              for (k=kk; k<=cbv_5; k=k+8) {
                y[k]=y[k]+A[i*ny+k]*tmp[i];
                y[(k+1)]=y[(k+1)]+A[i*ny+k+1]*tmp[i];
                y[(k+2)]=y[(k+2)]+A[i*ny+k+2]*tmp[i];
                y[(k+3)]=y[(k+3)]+A[i*ny+k+3]*tmp[i];
                y[(k+4)]=y[(k+4)]+A[i*ny+k+4]*tmp[i];
                y[(k+5)]=y[(k+5)]+A[i*ny+k+5]*tmp[i];
                y[(k+6)]=y[(k+6)]+A[i*ny+k+6]*tmp[i];
                y[(k+7)]=y[(k+7)]+A[i*ny+k+7]*tmp[i];
                y[k]=y[k]+A[(i+1)*ny+k]*tmp[(i+1)];
                y[(k+1)]=y[(k+1)]+A[(i+1)*ny+k+1]*tmp[(i+1)];
                y[(k+2)]=y[(k+2)]+A[(i+1)*ny+k+2]*tmp[(i+1)];
                y[(k+3)]=y[(k+3)]+A[(i+1)*ny+k+3]*tmp[(i+1)];
                y[(k+4)]=y[(k+4)]+A[(i+1)*ny+k+4]*tmp[(i+1)];
                y[(k+5)]=y[(k+5)]+A[(i+1)*ny+k+5]*tmp[(i+1)];
                y[(k+6)]=y[(k+6)]+A[(i+1)*ny+k+6]*tmp[(i+1)];
                y[(k+7)]=y[(k+7)]+A[(i+1)*ny+k+7]*tmp[(i+1)];
                y[k]=y[k]+A[(i+2)*ny+k]*tmp[(i+2)];
                y[(k+1)]=y[(k+1)]+A[(i+2)*ny+k+1]*tmp[(i+2)];
                y[(k+2)]=y[(k+2)]+A[(i+2)*ny+k+2]*tmp[(i+2)];
                y[(k+3)]=y[(k+3)]+A[(i+2)*ny+k+3]*tmp[(i+2)];
                y[(k+4)]=y[(k+4)]+A[(i+2)*ny+k+4]*tmp[(i+2)];
                y[(k+5)]=y[(k+5)]+A[(i+2)*ny+k+5]*tmp[(i+2)];
                y[(k+6)]=y[(k+6)]+A[(i+2)*ny+k+6]*tmp[(i+2)];
                y[(k+7)]=y[(k+7)]+A[(i+2)*ny+k+7]*tmp[(i+2)];
                y[k]=y[k]+A[(i+3)*ny+k]*tmp[(i+3)];
                y[(k+1)]=y[(k+1)]+A[(i+3)*ny+k+1]*tmp[(i+3)];
                y[(k+2)]=y[(k+2)]+A[(i+3)*ny+k+2]*tmp[(i+3)];
                y[(k+3)]=y[(k+3)]+A[(i+3)*ny+k+3]*tmp[(i+3)];
                y[(k+4)]=y[(k+4)]+A[(i+3)*ny+k+4]*tmp[(i+3)];
                y[(k+5)]=y[(k+5)]+A[(i+3)*ny+k+5]*tmp[(i+3)];
                y[(k+6)]=y[(k+6)]+A[(i+3)*ny+k+6]*tmp[(i+3)];
                y[(k+7)]=y[(k+7)]+A[(i+3)*ny+k+7]*tmp[(i+3)];
                y[k]=y[k]+A[(i+4)*ny+k]*tmp[(i+4)];
                y[(k+1)]=y[(k+1)]+A[(i+4)*ny+k+1]*tmp[(i+4)];
                y[(k+2)]=y[(k+2)]+A[(i+4)*ny+k+2]*tmp[(i+4)];
                y[(k+3)]=y[(k+3)]+A[(i+4)*ny+k+3]*tmp[(i+4)];
                y[(k+4)]=y[(k+4)]+A[(i+4)*ny+k+4]*tmp[(i+4)];
                y[(k+5)]=y[(k+5)]+A[(i+4)*ny+k+5]*tmp[(i+4)];
                y[(k+6)]=y[(k+6)]+A[(i+4)*ny+k+6]*tmp[(i+4)];
                y[(k+7)]=y[(k+7)]+A[(i+4)*ny+k+7]*tmp[(i+4)];
                y[k]=y[k]+A[(i+5)*ny+k]*tmp[(i+5)];
                y[(k+1)]=y[(k+1)]+A[(i+5)*ny+k+1]*tmp[(i+5)];
                y[(k+2)]=y[(k+2)]+A[(i+5)*ny+k+2]*tmp[(i+5)];
                y[(k+3)]=y[(k+3)]+A[(i+5)*ny+k+3]*tmp[(i+5)];
                y[(k+4)]=y[(k+4)]+A[(i+5)*ny+k+4]*tmp[(i+5)];
                y[(k+5)]=y[(k+5)]+A[(i+5)*ny+k+5]*tmp[(i+5)];
                y[(k+6)]=y[(k+6)]+A[(i+5)*ny+k+6]*tmp[(i+5)];
                y[(k+7)]=y[(k+7)]+A[(i+5)*ny+k+7]*tmp[(i+5)];
                y[k]=y[k]+A[(i+6)*ny+k]*tmp[(i+6)];
                y[(k+1)]=y[(k+1)]+A[(i+6)*ny+k+1]*tmp[(i+6)];
                y[(k+2)]=y[(k+2)]+A[(i+6)*ny+k+2]*tmp[(i+6)];
                y[(k+3)]=y[(k+3)]+A[(i+6)*ny+k+3]*tmp[(i+6)];
                y[(k+4)]=y[(k+4)]+A[(i+6)*ny+k+4]*tmp[(i+6)];
                y[(k+5)]=y[(k+5)]+A[(i+6)*ny+k+5]*tmp[(i+6)];
                y[(k+6)]=y[(k+6)]+A[(i+6)*ny+k+6]*tmp[(i+6)];
                y[(k+7)]=y[(k+7)]+A[(i+6)*ny+k+7]*tmp[(i+6)];
                y[k]=y[k]+A[(i+7)*ny+k]*tmp[(i+7)];
                y[(k+1)]=y[(k+1)]+A[(i+7)*ny+k+1]*tmp[(i+7)];
                y[(k+2)]=y[(k+2)]+A[(i+7)*ny+k+2]*tmp[(i+7)];
                y[(k+3)]=y[(k+3)]+A[(i+7)*ny+k+3]*tmp[(i+7)];
                y[(k+4)]=y[(k+4)]+A[(i+7)*ny+k+4]*tmp[(i+7)];
                y[(k+5)]=y[(k+5)]+A[(i+7)*ny+k+5]*tmp[(i+7)];
                y[(k+6)]=y[(k+6)]+A[(i+7)*ny+k+6]*tmp[(i+7)];
                y[(k+7)]=y[(k+7)]+A[(i+7)*ny+k+7]*tmp[(i+7)];
              }
              register int cbv_6;
              cbv_6=min(ny-1,kk+31);
#pragma ivdep
#pragma vector always
              for (; k<=cbv_6; k=k+1) {
                y[k]=y[k]+A[i*ny+k]*tmp[i];
                y[k]=y[k]+A[(i+1)*ny+k]*tmp[(i+1)];
                y[k]=y[k]+A[(i+2)*ny+k]*tmp[(i+2)];
                y[k]=y[k]+A[(i+3)*ny+k]*tmp[(i+3)];
                y[k]=y[k]+A[(i+4)*ny+k]*tmp[(i+4)];
                y[k]=y[k]+A[(i+5)*ny+k]*tmp[(i+5)];
                y[k]=y[k]+A[(i+6)*ny+k]*tmp[(i+6)];
                y[k]=y[k]+A[(i+7)*ny+k]*tmp[(i+7)];
              }
            }
          }
        }
        for (; i<=min(nx-1,ii+255); i=i+1) {
          tmp[i]=0;
          for (jjj=0; jjj<=ny-1; jjj=jjj+1024) 
            for (jj=jjj; jj<=min(ny-1,jjj+896); jj=jj+128) {
              register int cbv_7;
              cbv_7=min(ny-1,jj+127);
#pragma ivdep
#pragma vector always
              for (j=jj; j<=cbv_7; j=j+1) 
                tmp[i]=tmp[i]+A[i*ny+j]*x[j];
            }
          for (kkk=0; kkk<=ny-1; kkk=kkk+64) 
            for (kk=kkk; kk<=min(ny-1,kkk+32); kk=kk+32) {
              register int cbv_8;
              cbv_8=min(ny-1,kk+31)-7;
#pragma ivdep
#pragma vector always
              for (k=kk; k<=cbv_8; k=k+8) {
                y[k]=y[k]+A[i*ny+k]*tmp[i];
                y[(k+1)]=y[(k+1)]+A[i*ny+k+1]*tmp[i];
                y[(k+2)]=y[(k+2)]+A[i*ny+k+2]*tmp[i];
                y[(k+3)]=y[(k+3)]+A[i*ny+k+3]*tmp[i];
                y[(k+4)]=y[(k+4)]+A[i*ny+k+4]*tmp[i];
                y[(k+5)]=y[(k+5)]+A[i*ny+k+5]*tmp[i];
                y[(k+6)]=y[(k+6)]+A[i*ny+k+6]*tmp[i];
                y[(k+7)]=y[(k+7)]+A[i*ny+k+7]*tmp[i];
              }
              register int cbv_9;
              cbv_9=min(ny-1,kk+31);
#pragma ivdep
#pragma vector always
              for (; k<=cbv_9; k=k+1) 
                y[k]=y[k]+A[i*ny+k]*tmp[i];
            }
        }
      }
    }
}
/*@ end @*/



    orio_t_end = getClock();
    orio_t = orio_t_end - orio_t_start;
    if (orio_t < orio_t_min) orio_t_min = orio_t;
  }
  
  printf("{'[5, 4, 2, 3, 5, 1, 0, 0, 1, 2, 0, 2, 2, 0, 0, 0, 0, 1, 0]' : %g}", orio_t_min);

  

  return 0;
}

