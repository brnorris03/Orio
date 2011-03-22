/*@ begin PerfTuning (  
  def build
  {
  arg build_command = 'icc -O3 -openmp -lm -DDYNAMIC'; 
  }
  
  def performance_counter  
  { 
  arg repetitions = 5;
  }
  
  def performance_params 
  {
  param U1[] = [3];
  param U2[] = [4];

  param PAR[] = [True];
  param SCR[] = [True];
  param VEC[] = [False];
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

{
double* tmp=(double*)malloc(n*sizeof(double));
int i,j;
  
/*@ begin Loop(
  transform Composite(
    scalarreplace = (SCR, 'double'),
    vector = (VEC, ['ivdep','vector always'])
  )
  transform UnrollJam(ufactor=U1, parallelize=PAR)
  for (i=0; i<=n-1; i++) {
    tmp[i] = 0;
    y[i] = 0;
    transform UnrollJam(ufactor=U2)
    for (j=0; j<=n-1; j++) {
      tmp[i] = A[i*n+j]*x[j] + tmp[i];
      y[i] = B[i*n+j]*x[j] + y[i];
    }
    y[i] = a*tmp[i] + b*y[i];
  }
) @*/
/*@ end @*/
}

/*@ end @*/
  


