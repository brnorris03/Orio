/*@ begin PerfTuning (  
  def build
  {
  arg build_command = 'gcc -O3 -fopenmp -lm'; 
  }
  
  def performance_counter  
  { 
  arg repetitions = 5;
  }
  
  def performance_params 
  {
  param U1[]   = [5];
  param U2i[]  = [24];
  param U2ja[] = [1];
  param U2jb[] = [1];

  param PAR1[] = [False];
  param PAR2[] = [True];
  param SCR[]  = [True];
  param VEC1[] = [True];
  param VEC2[] = [False];
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
      tmp[i] = tmp[i] + A[i][j]*x[j];
    transform UnrollJam(ufactor=U2jb)
    for (j = 0; j<=ny-1; j++) 
      y[j] = y[j] + A[i][j]*tmp[i];
  }
) @*/
/*@ end @*/

/*@ end @*/
  


