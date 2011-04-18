

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define VSIZE 100
#define OSIZE 10
int V=100;
int O=10;
double **A2;
double ****T;
double ****R;
void malloc_arrays() {
  int i1,i2,i3,i4;
  A2 = (double**) malloc((V) * sizeof(double*));
  for (i1=0; i1<V; i1++) {
   A2[i1] = (double*) malloc((O) * sizeof(double));
  }
  T = (double****) malloc((V) * sizeof(double***));
  for (i1=0; i1<V; i1++) {
   T[i1] = (double***) malloc((O) * sizeof(double**));
   for (i2=0; i2<O; i2++) {
    T[i1][i2] = (double**) malloc((O) * sizeof(double*));
    for (i3=0; i3<O; i3++) {
     T[i1][i2][i3] = (double*) malloc((O) * sizeof(double));
  }}}
  R = (double****) malloc((V) * sizeof(double***));
  for (i1=0; i1<V; i1++) {
   R[i1] = (double***) malloc((V) * sizeof(double**));
   for (i2=0; i2<V; i2++) {
    R[i1][i2] = (double**) malloc((O) * sizeof(double*));
    for (i3=0; i3<O; i3++) {
     R[i1][i2][i3] = (double*) malloc((O) * sizeof(double));
  }}}
}

void init_input_vars() {
  int i1,i2,i3,i4;
  V = 100;
  O = 10;
  for (i1=0; i1<V; i1++)
   for (i2=0; i2<O; i2++)
    A2[i1][i2] = (i1+i2) % 5 + 1;
  for (i1=0; i1<V; i1++)
   for (i2=0; i2<O; i2++)
    for (i3=0; i3<O; i3++)
     for (i4=0; i4<O; i4++)
      T[i1][i2][i3][i4] = (i1+i2+i3+i4) % 5 + 1;
  for (i1=0; i1<V; i1++)
   for (i2=0; i2<V; i2++)
    for (i3=0; i3<O; i3++)
     for (i4=0; i4<O; i4++)
      R[i1][i2][i3][i4] = 0;
}



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
  malloc_arrays();
init_input_vars();


  double orio_t_start, orio_t_end, orio_t_total=0;
  int orio_i;

  for (orio_i=0; orio_i<REPS; orio_i++)
  {
    orio_t_start = getClock();
    
    

int v1,v2,o1,o2,ox;
int tv1,tv2,to1,to2,tox;


/*@ begin Loop(

  transform UnrollJam(ufactor=U_V1)
  for(v1=0; v1<=V-1; v1++) 
    transform UnrollJam(ufactor=U_V2)
    for(v2=0; v2<=V-1; v2++) 
      transform UnrollJam(ufactor=U_O1)
      for(o1=0; o1<=O-1; o1++) 
        transform UnrollJam(ufactor=U_O2)
        for(o2=0; o2<=O-1; o2++) 
	  transform UnrollJam(ufactor=U_OX)
	  for(ox=0; ox<=O-1; ox++) 
	    R[v1][v2][o1][o2] = R[v1][v2][o1][o2] + T[v1][ox][o1][o2] * A2[v2][ox];

) @*/
for (v1=0; v1<=V-1; v1=v1+1) 
  for (v2=0; v2<=V-1; v2=v2+1) 
    for (o1=0; o1<=O-1; o1=o1+1) 
      for (o2=0; o2<=O-1; o2=o2+1) 
        for (ox=0; ox<=O-1; ox=ox+1) 
          R[v1][v2][o1][o2]=R[v1][v2][o1][o2]+T[v1][ox][o1][o2]*A2[v2][ox];
/*@ end @*/


    orio_t_end = getClock();
    orio_t_total += orio_t_end - orio_t_start;
    printf("try: %g\n", orio_t_end - orio_t_start);
  }
  orio_t_total = orio_t_total / REPS;
  
  printf("{'[0, 0, 0, 0, 0]' : %g}", orio_t_total);

  

  return 0;
}

