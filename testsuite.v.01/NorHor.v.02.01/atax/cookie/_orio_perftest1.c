

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define min(a,b) ((a)<(b))?a:b
#define N 10000
#include "decl.h"

#include "init.c"



#ifdef BGP_COUNTER
#define SPRN_TBRL 0x10C // Time Base Read Lower Register (user & sup R/O)
#define SPRN_TBRU 0x10D // Time Base Read Upper Register (user & sup R/O)
#define _bgp_mfspr( SPRN )\
({\
  unsigned int tmp;\
  do {\
    asm volatile ("mfspr %0,%1" : "=&r" (tmp) : "i" (SPRN) : "memory" );\
  }\
  while(0);\
  tmp;\
})\

double getClock()
{
  union {
    unsigned int ul[2];
    unsigned long long ull;
  }
  hack;
  unsigned int utmp;
  do {
    utmp = _bgp_mfspr( SPRN_TBRU );
    hack.ul[1] = _bgp_mfspr( SPRN_TBRL );
    hack.ul[0] = _bgp_mfspr( SPRN_TBRU );
  }
  while(utmp != hack.ul[0]);
  return((double) hack.ull );
}
#else
double getClock()
{
  struct timezone tzp;
  struct timeval tp;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}
#endif

int main(int argc, char *argv[])
{
  init_input_vars();


  double orio_t_start, orio_t_end, orio_t_total=0;
  int orio_i;

  for (orio_i=0; orio_i<REPS; orio_i++)
  {
    orio_t_start = getClock();
    
    

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

int i,j;
double* tmp=(double*) malloc(nx*sizeof(double));
  
/*@ begin Loop(
  transform Composite(
    vector = (VEC1, ['ivdep','vector always'])
  )
  transform UnrollJam(ufactor=U1, parallelize=PAR1)
  for (i= 0; i<=ny-1; i++)
    y[i] = 0.0;

  transform Composite(
    scalarreplace = (SCR, 'double'),
    vector = (VEC2, ['ivdep','vector always'])
  )
  transform UnrollJam(ufactor=U2i, parallelize=PAR2)
  for (i = 0; i<=nx-1; i++) {
    tmp[i] = 0;
    transform UnrollJam(ufactor=U2ja)
    for (j = 0; j<=ny-1; j++) 
      tmp[i] = tmp[i] + A[i*ny+j]*x[j];
    transform UnrollJam(ufactor=U2jb)
    for (j = 0; j<=ny-1; j++) 
      y[j] = y[j] + A[i*ny+j]*tmp[i];
  }
) @*/
{
  register int cbv_1;
  cbv_1=ny-20;
#pragma ivdep
#pragma vector always
  for (i=0; i<=cbv_1; i=i+20) {
    y[i]=0.0;
    y[(i+1)]=0.0;
    y[(i+2)]=0.0;
    y[(i+3)]=0.0;
    y[(i+4)]=0.0;
    y[(i+5)]=0.0;
    y[(i+6)]=0.0;
    y[(i+7)]=0.0;
    y[(i+8)]=0.0;
    y[(i+9)]=0.0;
    y[(i+10)]=0.0;
    y[(i+11)]=0.0;
    y[(i+12)]=0.0;
    y[(i+13)]=0.0;
    y[(i+14)]=0.0;
    y[(i+15)]=0.0;
    y[(i+16)]=0.0;
    y[(i+17)]=0.0;
    y[(i+18)]=0.0;
    y[(i+19)]=0.0;
  }
  register int cbv_2;
  cbv_2=ny-1;
#pragma ivdep
#pragma vector always
  for (; i<=cbv_2; i=i+1) 
    y[i]=0.0;
}
{
  for (i=0; i<=nx-8; i=i+8) {
    tmp[i]=0;
    {
      for (j=0; j<=ny-17; j=j+17) {
        tmp[i]=tmp[i]+A[i*ny+j]*x[j];
        tmp[i]=tmp[i]+A[i*ny+j+1]*x[(j+1)];
        tmp[i]=tmp[i]+A[i*ny+j+2]*x[(j+2)];
        tmp[i]=tmp[i]+A[i*ny+j+3]*x[(j+3)];
        tmp[i]=tmp[i]+A[i*ny+j+4]*x[(j+4)];
        tmp[i]=tmp[i]+A[i*ny+j+5]*x[(j+5)];
        tmp[i]=tmp[i]+A[i*ny+j+6]*x[(j+6)];
        tmp[i]=tmp[i]+A[i*ny+j+7]*x[(j+7)];
        tmp[i]=tmp[i]+A[i*ny+j+8]*x[(j+8)];
        tmp[i]=tmp[i]+A[i*ny+j+9]*x[(j+9)];
        tmp[i]=tmp[i]+A[i*ny+j+10]*x[(j+10)];
        tmp[i]=tmp[i]+A[i*ny+j+11]*x[(j+11)];
        tmp[i]=tmp[i]+A[i*ny+j+12]*x[(j+12)];
        tmp[i]=tmp[i]+A[i*ny+j+13]*x[(j+13)];
        tmp[i]=tmp[i]+A[i*ny+j+14]*x[(j+14)];
        tmp[i]=tmp[i]+A[i*ny+j+15]*x[(j+15)];
        tmp[i]=tmp[i]+A[i*ny+j+16]*x[(j+16)];
      }
      for (; j<=ny-1; j=j+1) 
        tmp[i]=tmp[i]+A[i*ny+j]*x[j];
    }
    {
      for (j=0; j<=ny-10; j=j+10) {
        y[j]=y[j]+A[i*ny+j]*tmp[i];
        y[(j+1)]=y[(j+1)]+A[i*ny+j+1]*tmp[i];
        y[(j+2)]=y[(j+2)]+A[i*ny+j+2]*tmp[i];
        y[(j+3)]=y[(j+3)]+A[i*ny+j+3]*tmp[i];
        y[(j+4)]=y[(j+4)]+A[i*ny+j+4]*tmp[i];
        y[(j+5)]=y[(j+5)]+A[i*ny+j+5]*tmp[i];
        y[(j+6)]=y[(j+6)]+A[i*ny+j+6]*tmp[i];
        y[(j+7)]=y[(j+7)]+A[i*ny+j+7]*tmp[i];
        y[(j+8)]=y[(j+8)]+A[i*ny+j+8]*tmp[i];
        y[(j+9)]=y[(j+9)]+A[i*ny+j+9]*tmp[i];
      }
      for (; j<=ny-1; j=j+1) 
        y[j]=y[j]+A[i*ny+j]*tmp[i];
    }
    tmp[(i+1)]=0;
    {
      for (j=0; j<=ny-17; j=j+17) {
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j]*x[j];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+1]*x[(j+1)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+2]*x[(j+2)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+3]*x[(j+3)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+4]*x[(j+4)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+5]*x[(j+5)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+6]*x[(j+6)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+7]*x[(j+7)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+8]*x[(j+8)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+9]*x[(j+9)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+10]*x[(j+10)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+11]*x[(j+11)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+12]*x[(j+12)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+13]*x[(j+13)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+14]*x[(j+14)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+15]*x[(j+15)];
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j+16]*x[(j+16)];
      }
      for (; j<=ny-1; j=j+1) 
        tmp[(i+1)]=tmp[(i+1)]+A[(i+1)*ny+j]*x[j];
    }
    {
      for (j=0; j<=ny-10; j=j+10) {
        y[j]=y[j]+A[(i+1)*ny+j]*tmp[(i+1)];
        y[(j+1)]=y[(j+1)]+A[(i+1)*ny+j+1]*tmp[(i+1)];
        y[(j+2)]=y[(j+2)]+A[(i+1)*ny+j+2]*tmp[(i+1)];
        y[(j+3)]=y[(j+3)]+A[(i+1)*ny+j+3]*tmp[(i+1)];
        y[(j+4)]=y[(j+4)]+A[(i+1)*ny+j+4]*tmp[(i+1)];
        y[(j+5)]=y[(j+5)]+A[(i+1)*ny+j+5]*tmp[(i+1)];
        y[(j+6)]=y[(j+6)]+A[(i+1)*ny+j+6]*tmp[(i+1)];
        y[(j+7)]=y[(j+7)]+A[(i+1)*ny+j+7]*tmp[(i+1)];
        y[(j+8)]=y[(j+8)]+A[(i+1)*ny+j+8]*tmp[(i+1)];
        y[(j+9)]=y[(j+9)]+A[(i+1)*ny+j+9]*tmp[(i+1)];
      }
      for (; j<=ny-1; j=j+1) 
        y[j]=y[j]+A[(i+1)*ny+j]*tmp[(i+1)];
    }
    tmp[(i+2)]=0;
    {
      for (j=0; j<=ny-17; j=j+17) {
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j]*x[j];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+1]*x[(j+1)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+2]*x[(j+2)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+3]*x[(j+3)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+4]*x[(j+4)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+5]*x[(j+5)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+6]*x[(j+6)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+7]*x[(j+7)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+8]*x[(j+8)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+9]*x[(j+9)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+10]*x[(j+10)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+11]*x[(j+11)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+12]*x[(j+12)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+13]*x[(j+13)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+14]*x[(j+14)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+15]*x[(j+15)];
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j+16]*x[(j+16)];
      }
      for (; j<=ny-1; j=j+1) 
        tmp[(i+2)]=tmp[(i+2)]+A[(i+2)*ny+j]*x[j];
    }
    {
      for (j=0; j<=ny-10; j=j+10) {
        y[j]=y[j]+A[(i+2)*ny+j]*tmp[(i+2)];
        y[(j+1)]=y[(j+1)]+A[(i+2)*ny+j+1]*tmp[(i+2)];
        y[(j+2)]=y[(j+2)]+A[(i+2)*ny+j+2]*tmp[(i+2)];
        y[(j+3)]=y[(j+3)]+A[(i+2)*ny+j+3]*tmp[(i+2)];
        y[(j+4)]=y[(j+4)]+A[(i+2)*ny+j+4]*tmp[(i+2)];
        y[(j+5)]=y[(j+5)]+A[(i+2)*ny+j+5]*tmp[(i+2)];
        y[(j+6)]=y[(j+6)]+A[(i+2)*ny+j+6]*tmp[(i+2)];
        y[(j+7)]=y[(j+7)]+A[(i+2)*ny+j+7]*tmp[(i+2)];
        y[(j+8)]=y[(j+8)]+A[(i+2)*ny+j+8]*tmp[(i+2)];
        y[(j+9)]=y[(j+9)]+A[(i+2)*ny+j+9]*tmp[(i+2)];
      }
      for (; j<=ny-1; j=j+1) 
        y[j]=y[j]+A[(i+2)*ny+j]*tmp[(i+2)];
    }
    tmp[(i+3)]=0;
    {
      for (j=0; j<=ny-17; j=j+17) {
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j]*x[j];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+1]*x[(j+1)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+2]*x[(j+2)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+3]*x[(j+3)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+4]*x[(j+4)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+5]*x[(j+5)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+6]*x[(j+6)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+7]*x[(j+7)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+8]*x[(j+8)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+9]*x[(j+9)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+10]*x[(j+10)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+11]*x[(j+11)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+12]*x[(j+12)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+13]*x[(j+13)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+14]*x[(j+14)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+15]*x[(j+15)];
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j+16]*x[(j+16)];
      }
      for (; j<=ny-1; j=j+1) 
        tmp[(i+3)]=tmp[(i+3)]+A[(i+3)*ny+j]*x[j];
    }
    {
      for (j=0; j<=ny-10; j=j+10) {
        y[j]=y[j]+A[(i+3)*ny+j]*tmp[(i+3)];
        y[(j+1)]=y[(j+1)]+A[(i+3)*ny+j+1]*tmp[(i+3)];
        y[(j+2)]=y[(j+2)]+A[(i+3)*ny+j+2]*tmp[(i+3)];
        y[(j+3)]=y[(j+3)]+A[(i+3)*ny+j+3]*tmp[(i+3)];
        y[(j+4)]=y[(j+4)]+A[(i+3)*ny+j+4]*tmp[(i+3)];
        y[(j+5)]=y[(j+5)]+A[(i+3)*ny+j+5]*tmp[(i+3)];
        y[(j+6)]=y[(j+6)]+A[(i+3)*ny+j+6]*tmp[(i+3)];
        y[(j+7)]=y[(j+7)]+A[(i+3)*ny+j+7]*tmp[(i+3)];
        y[(j+8)]=y[(j+8)]+A[(i+3)*ny+j+8]*tmp[(i+3)];
        y[(j+9)]=y[(j+9)]+A[(i+3)*ny+j+9]*tmp[(i+3)];
      }
      for (; j<=ny-1; j=j+1) 
        y[j]=y[j]+A[(i+3)*ny+j]*tmp[(i+3)];
    }
    tmp[(i+4)]=0;
    {
      for (j=0; j<=ny-17; j=j+17) {
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j]*x[j];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+1]*x[(j+1)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+2]*x[(j+2)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+3]*x[(j+3)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+4]*x[(j+4)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+5]*x[(j+5)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+6]*x[(j+6)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+7]*x[(j+7)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+8]*x[(j+8)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+9]*x[(j+9)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+10]*x[(j+10)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+11]*x[(j+11)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+12]*x[(j+12)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+13]*x[(j+13)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+14]*x[(j+14)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+15]*x[(j+15)];
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j+16]*x[(j+16)];
      }
      for (; j<=ny-1; j=j+1) 
        tmp[(i+4)]=tmp[(i+4)]+A[(i+4)*ny+j]*x[j];
    }
    {
      for (j=0; j<=ny-10; j=j+10) {
        y[j]=y[j]+A[(i+4)*ny+j]*tmp[(i+4)];
        y[(j+1)]=y[(j+1)]+A[(i+4)*ny+j+1]*tmp[(i+4)];
        y[(j+2)]=y[(j+2)]+A[(i+4)*ny+j+2]*tmp[(i+4)];
        y[(j+3)]=y[(j+3)]+A[(i+4)*ny+j+3]*tmp[(i+4)];
        y[(j+4)]=y[(j+4)]+A[(i+4)*ny+j+4]*tmp[(i+4)];
        y[(j+5)]=y[(j+5)]+A[(i+4)*ny+j+5]*tmp[(i+4)];
        y[(j+6)]=y[(j+6)]+A[(i+4)*ny+j+6]*tmp[(i+4)];
        y[(j+7)]=y[(j+7)]+A[(i+4)*ny+j+7]*tmp[(i+4)];
        y[(j+8)]=y[(j+8)]+A[(i+4)*ny+j+8]*tmp[(i+4)];
        y[(j+9)]=y[(j+9)]+A[(i+4)*ny+j+9]*tmp[(i+4)];
      }
      for (; j<=ny-1; j=j+1) 
        y[j]=y[j]+A[(i+4)*ny+j]*tmp[(i+4)];
    }
    tmp[(i+5)]=0;
    {
      for (j=0; j<=ny-17; j=j+17) {
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j]*x[j];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+1]*x[(j+1)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+2]*x[(j+2)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+3]*x[(j+3)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+4]*x[(j+4)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+5]*x[(j+5)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+6]*x[(j+6)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+7]*x[(j+7)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+8]*x[(j+8)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+9]*x[(j+9)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+10]*x[(j+10)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+11]*x[(j+11)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+12]*x[(j+12)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+13]*x[(j+13)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+14]*x[(j+14)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+15]*x[(j+15)];
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j+16]*x[(j+16)];
      }
      for (; j<=ny-1; j=j+1) 
        tmp[(i+5)]=tmp[(i+5)]+A[(i+5)*ny+j]*x[j];
    }
    {
      for (j=0; j<=ny-10; j=j+10) {
        y[j]=y[j]+A[(i+5)*ny+j]*tmp[(i+5)];
        y[(j+1)]=y[(j+1)]+A[(i+5)*ny+j+1]*tmp[(i+5)];
        y[(j+2)]=y[(j+2)]+A[(i+5)*ny+j+2]*tmp[(i+5)];
        y[(j+3)]=y[(j+3)]+A[(i+5)*ny+j+3]*tmp[(i+5)];
        y[(j+4)]=y[(j+4)]+A[(i+5)*ny+j+4]*tmp[(i+5)];
        y[(j+5)]=y[(j+5)]+A[(i+5)*ny+j+5]*tmp[(i+5)];
        y[(j+6)]=y[(j+6)]+A[(i+5)*ny+j+6]*tmp[(i+5)];
        y[(j+7)]=y[(j+7)]+A[(i+5)*ny+j+7]*tmp[(i+5)];
        y[(j+8)]=y[(j+8)]+A[(i+5)*ny+j+8]*tmp[(i+5)];
        y[(j+9)]=y[(j+9)]+A[(i+5)*ny+j+9]*tmp[(i+5)];
      }
      for (; j<=ny-1; j=j+1) 
        y[j]=y[j]+A[(i+5)*ny+j]*tmp[(i+5)];
    }
    tmp[(i+6)]=0;
    {
      for (j=0; j<=ny-17; j=j+17) {
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j]*x[j];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+1]*x[(j+1)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+2]*x[(j+2)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+3]*x[(j+3)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+4]*x[(j+4)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+5]*x[(j+5)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+6]*x[(j+6)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+7]*x[(j+7)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+8]*x[(j+8)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+9]*x[(j+9)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+10]*x[(j+10)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+11]*x[(j+11)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+12]*x[(j+12)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+13]*x[(j+13)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+14]*x[(j+14)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+15]*x[(j+15)];
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j+16]*x[(j+16)];
      }
      for (; j<=ny-1; j=j+1) 
        tmp[(i+6)]=tmp[(i+6)]+A[(i+6)*ny+j]*x[j];
    }
    {
      for (j=0; j<=ny-10; j=j+10) {
        y[j]=y[j]+A[(i+6)*ny+j]*tmp[(i+6)];
        y[(j+1)]=y[(j+1)]+A[(i+6)*ny+j+1]*tmp[(i+6)];
        y[(j+2)]=y[(j+2)]+A[(i+6)*ny+j+2]*tmp[(i+6)];
        y[(j+3)]=y[(j+3)]+A[(i+6)*ny+j+3]*tmp[(i+6)];
        y[(j+4)]=y[(j+4)]+A[(i+6)*ny+j+4]*tmp[(i+6)];
        y[(j+5)]=y[(j+5)]+A[(i+6)*ny+j+5]*tmp[(i+6)];
        y[(j+6)]=y[(j+6)]+A[(i+6)*ny+j+6]*tmp[(i+6)];
        y[(j+7)]=y[(j+7)]+A[(i+6)*ny+j+7]*tmp[(i+6)];
        y[(j+8)]=y[(j+8)]+A[(i+6)*ny+j+8]*tmp[(i+6)];
        y[(j+9)]=y[(j+9)]+A[(i+6)*ny+j+9]*tmp[(i+6)];
      }
      for (; j<=ny-1; j=j+1) 
        y[j]=y[j]+A[(i+6)*ny+j]*tmp[(i+6)];
    }
    tmp[(i+7)]=0;
    {
      for (j=0; j<=ny-17; j=j+17) {
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j]*x[j];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+1]*x[(j+1)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+2]*x[(j+2)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+3]*x[(j+3)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+4]*x[(j+4)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+5]*x[(j+5)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+6]*x[(j+6)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+7]*x[(j+7)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+8]*x[(j+8)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+9]*x[(j+9)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+10]*x[(j+10)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+11]*x[(j+11)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+12]*x[(j+12)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+13]*x[(j+13)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+14]*x[(j+14)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+15]*x[(j+15)];
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j+16]*x[(j+16)];
      }
      for (; j<=ny-1; j=j+1) 
        tmp[(i+7)]=tmp[(i+7)]+A[(i+7)*ny+j]*x[j];
    }
    {
      for (j=0; j<=ny-10; j=j+10) {
        y[j]=y[j]+A[(i+7)*ny+j]*tmp[(i+7)];
        y[(j+1)]=y[(j+1)]+A[(i+7)*ny+j+1]*tmp[(i+7)];
        y[(j+2)]=y[(j+2)]+A[(i+7)*ny+j+2]*tmp[(i+7)];
        y[(j+3)]=y[(j+3)]+A[(i+7)*ny+j+3]*tmp[(i+7)];
        y[(j+4)]=y[(j+4)]+A[(i+7)*ny+j+4]*tmp[(i+7)];
        y[(j+5)]=y[(j+5)]+A[(i+7)*ny+j+5]*tmp[(i+7)];
        y[(j+6)]=y[(j+6)]+A[(i+7)*ny+j+6]*tmp[(i+7)];
        y[(j+7)]=y[(j+7)]+A[(i+7)*ny+j+7]*tmp[(i+7)];
        y[(j+8)]=y[(j+8)]+A[(i+7)*ny+j+8]*tmp[(i+7)];
        y[(j+9)]=y[(j+9)]+A[(i+7)*ny+j+9]*tmp[(i+7)];
      }
      for (; j<=ny-1; j=j+1) 
        y[j]=y[j]+A[(i+7)*ny+j]*tmp[(i+7)];
    }
  }
  for (; i<=nx-1; i=i+1) {
    tmp[i]=0;
    {
      for (j=0; j<=ny-17; j=j+17) {
        tmp[i]=tmp[i]+A[i*ny+j]*x[j];
        tmp[i]=tmp[i]+A[i*ny+j+1]*x[(j+1)];
        tmp[i]=tmp[i]+A[i*ny+j+2]*x[(j+2)];
        tmp[i]=tmp[i]+A[i*ny+j+3]*x[(j+3)];
        tmp[i]=tmp[i]+A[i*ny+j+4]*x[(j+4)];
        tmp[i]=tmp[i]+A[i*ny+j+5]*x[(j+5)];
        tmp[i]=tmp[i]+A[i*ny+j+6]*x[(j+6)];
        tmp[i]=tmp[i]+A[i*ny+j+7]*x[(j+7)];
        tmp[i]=tmp[i]+A[i*ny+j+8]*x[(j+8)];
        tmp[i]=tmp[i]+A[i*ny+j+9]*x[(j+9)];
        tmp[i]=tmp[i]+A[i*ny+j+10]*x[(j+10)];
        tmp[i]=tmp[i]+A[i*ny+j+11]*x[(j+11)];
        tmp[i]=tmp[i]+A[i*ny+j+12]*x[(j+12)];
        tmp[i]=tmp[i]+A[i*ny+j+13]*x[(j+13)];
        tmp[i]=tmp[i]+A[i*ny+j+14]*x[(j+14)];
        tmp[i]=tmp[i]+A[i*ny+j+15]*x[(j+15)];
        tmp[i]=tmp[i]+A[i*ny+j+16]*x[(j+16)];
      }
      for (; j<=ny-1; j=j+1) 
        tmp[i]=tmp[i]+A[i*ny+j]*x[j];
    }
    {
      for (j=0; j<=ny-10; j=j+10) {
        y[j]=y[j]+A[i*ny+j]*tmp[i];
        y[(j+1)]=y[(j+1)]+A[i*ny+j+1]*tmp[i];
        y[(j+2)]=y[(j+2)]+A[i*ny+j+2]*tmp[i];
        y[(j+3)]=y[(j+3)]+A[i*ny+j+3]*tmp[i];
        y[(j+4)]=y[(j+4)]+A[i*ny+j+4]*tmp[i];
        y[(j+5)]=y[(j+5)]+A[i*ny+j+5]*tmp[i];
        y[(j+6)]=y[(j+6)]+A[i*ny+j+6]*tmp[i];
        y[(j+7)]=y[(j+7)]+A[i*ny+j+7]*tmp[i];
        y[(j+8)]=y[(j+8)]+A[i*ny+j+8]*tmp[i];
        y[(j+9)]=y[(j+9)]+A[i*ny+j+9]*tmp[i];
      }
      for (; j<=ny-1; j=j+1) 
        y[j]=y[j]+A[i*ny+j]*tmp[i];
    }
  }
}
/*@ end @*/



    orio_t_end = getClock();
    orio_t_total += orio_t_end - orio_t_start;
    printf("try: %g\n", orio_t_end - orio_t_start);
  }
  orio_t_total = orio_t_total / REPS;
  
  printf("{'[19, 7, 16, 9, 0, 0, 0, 1, 0]' : %g}", orio_t_total);

  

  return 0;
}

