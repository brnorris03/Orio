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
  param U1[] = [1];
  param U2i[] = [12];
  param U2j[] = [1];

  param PAR1[] = [False];
  param PAR2[] = [True];
  param SCR[] = [False];
  param VEC1[] = [True];
  param VEC2[] = [True];
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
  
/*@ begin Loop(

  transform Composite(
    vector = (VEC1, ['ivdep','vector always'])
  )
  transform UnrollJam(ufactor=U1, parallelize=PAR1)
  for (i = 0; i <= ny-1; i++)
    s[i] = 0;

  transform Composite(
    scalarreplace = (SCR, 'double'),
    vector = (VEC2, ['ivdep','vector always'])
  )
  transform UnrollJam(ufactor=U2i, parallelize=PAR2)
  for (i = 0; i <= nx-1; i++) {
    q[i] = 0;
    transform UnrollJam(ufactor=U2j)
    for (j = 0; j <= ny-1; j++) {
      s[j] = s[j] + r[i]*A[i][j];
      q[i] = q[i] + A[i][j]*p[j];
    }
  }
) @*/
/*@ end @*/

/*@ end @*/
  


