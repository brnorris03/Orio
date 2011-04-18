/*@ begin PerfTuning (  
  def build
  {
  arg build_command = 'icc -O3 -openmp -lm -DDYNAMIC'; 
  }
  
  def performance_counter  
  { 
  arg repetitions = 35;
  }
  
  def performance_params 
  {
  param U1[] = range(1,31);
  param U2i[] = range(1,31);
  param U2j[] = range(1,31);

  param PAR1[] = [False,True];
  param PAR2[] = [False,True];
  param SCR[] = [False,True];
  param VEC1[] = [False,True];
  param VEC2[] = [False,True];
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
      s[j] = s[j] + r[i]*A[i*ny+j];
      q[i] = q[i] + A[i*ny+j]*p[j];
    }
  }
) @*/
/*@ end @*/

/*@ end @*/
  


