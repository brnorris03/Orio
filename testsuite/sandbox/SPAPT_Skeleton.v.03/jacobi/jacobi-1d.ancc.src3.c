/*@ begin PerfTuning (          
  def build  
  {  
    arg build_command = 'gcc -O3 -fopenmp ';
    arg libs = '-lm -lrt';
  }  
     
  def performance_counter           
  {  
    arg repetitions = 35;  
  } 
 
  def performance_params  
  { 

    param T2_I[] = [];
    param T2_J[] = [];
    param T2_Ia[] = [];
    param T2_Ja[] = [];

    param U2_I[]  = [];
    param U2_J[]  = [];

    param RT2_I[] = [];
    param RT2_J[] = [];

    param SCR[]  = [];
    param VEC2[] = [];
    param OMP[] = [];


    constraint tileI2 = ((T2_Ia == 1) or (T2_Ia % T2_I == 0));
    constraint tileJ2 = ((T2_Ja == 1) or (T2_Ja % T2_J == 0));
    constraint reg_capacity = (RT2_I*RT2_J <= 150);
    constraint unroll_limit = ((U2_I == 1) or (U2_J == 1));

  } 

 
  def input_params
  {
    param T[] = [40000];
    param N[] = [40000];
  }

  def input_vars
  { 
   arg decl_file = 'jacobi-1d_decl_code.h';
   arg init_file = 'jacobi-1d_init_code.c';
  } 

  def search  
  {
    arg algorithm = 'Randomsearch';  
    arg total_runs = 10000; 
  }
) @*/  


int i,j, k,t;
int it, jt, kt;
int ii, jj, kk;
int iii, jjj, kkk;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))
/*@ begin Loop(
transform Composite(
      tile = [('i',T2_I,'ii'),('j',T2_J,'jj'),
             (('ii','i'),T2_Ia,'iii'),(('jj','j'),T2_Ja,'jjj')],
      unrolljam = (['i','j'],[U2_I,U2_J]),
      scalarreplace = (SCR, 'double'),
      regtile = (['i','j'],[RT2_I,RT2_J]),
      vector = (VEC2, ['ivdep','vector always']),
      openmp = (OMP, 'omp parallel for private(iii,jjj,ii,jj,i,j)')
)
for (i=1; i<=T-1; i++) 
  for (j=1; j<=N-2; j++) 
    a[i][j] = 0.333 * (a[i-1][j-1] + a[i-1][j] + a[i-1][j+1]);

) @*/

/*@ end @*/
/*@ end @*/





