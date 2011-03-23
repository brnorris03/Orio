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
  param U1i[] = range(1,31);
  param U1j[] = range(1,31);
  param U1ia[] = [1,4,8,16,32];
  param U1ja[] = [1,4,8,16,32];
  param PAR1i[] = [False,True];
  param PAR1j[] = [False,True];
  param SCR[] = [False,True];
  param VEC[] = [False,True];
  }
			 
  def search 
  { 
    arg algorithm = 'Randomsearch'; 
    arg total_runs = 20;
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
 int i,j,it,jt;
  
/*@ begin Loop(
  transform Composite(
    scalarreplace = (SCR, 'double'),
    regtile = (['i','j'],[U1ia,U1ja]),
    vector = (VEC, ['ivdep','vector always'])
  )
  transform UnrollJam(ufactor=U1i, parallelize=PAR1i)
  for (i=0; i<=n-1; i++) {
    tmp[i] = 0;
    y[i] = 0;
    transform UnrollJam(ufactor=U1j, parallelize=PAR1j)
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
  


