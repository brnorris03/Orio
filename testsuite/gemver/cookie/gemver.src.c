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
  param U2[]  = [1];
  param U3[]  = [1];
  param U4i[] = [1];
  param U4j[] = [1];
  param U5[]  = [1];
  param U6i[] = [1];
  param U6j[] = [1];
  param U7[]  = [1];

  param PAR1[] = [False];
  param PAR2[] = [False];
  param PAR3[] = [False];
  param PAR4[] = [False];
  param PAR5[] = [False];
  param PAR6[] = [True];
  param PAR7[] = [False];

  param SCR4[] = [False];

  param VEC4[] = [True];
  param VEC6[] = [False];

  }
			 
  def search 
  { 
    arg algorithm = 'Exhaustive'; 
#    arg algorithm = 'Simplex'; 
#    arg total_runs = 1;
  } 
  
  def input_params 
  {
  let N = [10000];
  param Nx[] = N;
  param Ny[] = N;
  }
  
  def input_vars
  {
  arg decl_file = 'decl_code.h';
  arg init_file = 'init_code.c';
  }
) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

int i,j,ii,jj,it,jt;

/*@ begin Loop(

transform UnrollJam(ufactor=U1, parallelize=PAR1)
for (i=0;i<=min(nx-1,ny-1);i++) {
  x[i]=0; 
  w[i]=0;
}

transform UnrollJam(ufactor=U2, parallelize=PAR2)
for (i=nx;i<=ny-1;i++)
  x[i]=0;

transform UnrollJam(ufactor=U3, parallelize=PAR3)
for (i=ny;i<=nx-1;i++)
  w[i]=0;

transform Composite(
 scalarreplace = (SCR4, 'double'),
 vector = (VEC4, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U4i, parallelize=PAR4)
for (j=0;j<=nx-1;j++)
  transform UnrollJam(ufactor=U4j)
  for (i=0;i<=ny-1;i++) {
    B[j][i]=u2[j]*v2[i]+u1[j]*v1[i]+A[j][i];
    x[i]=y[j]*B[j][i]+x[i];
  }

transform UnrollJam(ufactor=U5, parallelize=PAR5)
for (i=0;i<=ny-1;i++)
  x[i]=z[i]+b*x[i];

transform Composite(
 vector = (VEC6, ['ivdep','vector always'])
)
transform UnrollJam(ufactor=U6i, parallelize=PAR6)
for (i=0;i<=nx-1;i++) 
  transform UnrollJam(ufactor=U6j)
  for (j=0;j<=ny-1;j++) 
    w[i]=B[i][j]*x[j]+w[i];

transform UnrollJam(ufactor=U7, parallelize=PAR7)
for (i=0;i<=nx-1;i++) 
  w[i]=a*w[i];

) @*/
/*@ end @*/

/*@ end @*/

