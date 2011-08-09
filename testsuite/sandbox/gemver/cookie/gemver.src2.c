/*@ begin PerfTuning (  
  def build
  {
  arg build_command = 'icc -O3 -openmp -DDYNAMIC'; 
  arg libs = '-lm -lrt';
  }
  
  def performance_counter  
  { 
  arg repetitions = 35;
  }
  
  def performance_params 
  {
  param U1[]  = range(1,31);
  param U2i[] = [1]; # MUST be 1, otherwise it generates INCORRECT results (need debugging later)
  param U2j[] = range(1,31);
  param U3[]  = range(1,31);
  param U4i[] = range(1,31);
  param U4j[] = range(1,31);

  param PAR1[] = [False,True];
  param PAR2[] = [False,True]; 
  param PAR3[] = [False,True];
  param PAR4[] = [False,True];

  param SCR1[] = [False,True]; 
  param SCR2[] = [False,True];

  param VEC1[] = [False,True]; 
  param VEC2[] = [False,True];
  param VEC3[] = [False,True];
  }
			 
  def search 
  { 
    arg algorithm = 'Randomsearch'; 
    arg total_runs = 5000;
  } 
  
  def input_params 
  {
  param N[] = [10000];
  }
  
  def input_vars
  {
  arg decl_file = 'decl.h';
  arg init_file = 'init.c';
  }
) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

int i,j,ii,jj,it,jt;

/*@ begin Loop(

transform UnrollJam(ufactor=U1, parallelize=PAR1)
for (i=0;i<=n-1;i++) {
  x[i]=0;
  w[i]=0;
 }

transform Composite(
 scalarreplace = (SCR1, 'double'),
 vector = (VEC1, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U2i, parallelize=PAR2)
for (j=0;j<=n-1;j++) {
  transform UnrollJam(ufactor=U2j)
  for (i=0;i<=n-1;i++) {
    B[j*n+i]=u2[j]*v2[i]+u1[j]*v1[i]+A[j*n+i];
    x[i]=y[j]*B[j*n+i]+x[i];
  }
 }

transform Composite(
 vector = (VEC2, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U3, parallelize=PAR3)
for (i=0;i<=n-1;i++) {
  x[i]=b*x[i]+z[i];
 }

transform Composite(
 scalarreplace = (SCR2, 'double'),
 vector = (VEC3, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U4i, parallelize=PAR4)
for (i = 0; i <= n-1; i++) {
  transform UnrollJam(ufactor=U4j)
  for (j = 0; j <= n-1; j++) {
    w[i] = w[i] + B[i*n+j]*x[j];
  }
  w[i] = a*w[i];
 }

) @*/
/*@ end @*/

/*@ end @*/



