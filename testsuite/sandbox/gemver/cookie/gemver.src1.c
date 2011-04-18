/*@ begin PerfTuning (  
  def build
  {
  arg build_command = 'icc -O3 -openmp -lm'; 
  }
  
  def performance_counter  
  { 
  arg repetitions = 5;
  }
  
  def performance_params 
  {
  param U1[]  = [1];
  param U2i[] = [1];
  param U2j[] = [1];
  param U3[]  = [1];
  param U4i[] = [1];
  param U4j[] = [1];

  param PAR1[] = [False];
  param PAR2[] = [False];
  param PAR3[] = [False];
  param PAR4[] = [True];

  param SCR1[] = [False];
  param SCR2[] = [True];

  param VEC1[] = [False];
  param VEC2[] = [False];
  param VEC3[] = [False];
  }
			 
  def search 
  { 
    arg algorithm = 'Exhaustive'; 
#    arg algorithm = 'Simplex'; 
#    arg total_runs = 1;
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
    B[j][i]=u2[j]*v2[i]+u1[j]*v1[i]+A[j][i];
    x[i]=y[j]*B[j][i]+x[i];
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
    w[i] = w[i] + B[i][j]*x[j];
  }
  w[i] = a*w[i];
 }

) @*/
/*@ end @*/

/*@ end @*/



