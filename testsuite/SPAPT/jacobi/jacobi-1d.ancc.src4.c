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

    param T2_I[] = [1,16,32,64,128,256,512];
    param T2_J[] = [1,16,32,64,128,256,512];
    param T2_Ia[] = [1,64,128,256,512,1024,2048];
    param T2_Ja[] = [1,64,128,256,512,1024,2048];

    param U_I[]  = range(1,31); 
    param U_J[]  = range(1,31); 

    param RT_I[] = [1,8,32];
    param RT_J[] = [1,8,32];

    constraint tileI2 = ((T2_Ia == 1) or (T2_Ia % T2_I == 0));
    constraint tileJ2 = ((T2_Ja == 1) or (T2_Ja % T2_J == 0));
    constraint reg_capacity = (RT_I*RT_J <= 150);
    constraint unroll_limit = ((U_I == 1) or (U_J == 1));

  } 
 
  def input_params
  {
    param T[] = [10000];
    param N[] = [10000];
  }

  def input_vars
  { 
   arg decl_file = 'jacobi-1d_decl_code.h';
   arg init_file = 'jacobi-1d_init_code.c';
  } 

  def search  
  {
    arg algorithm = 'Randomsearch';  
    arg total_runs = 100000; 
  }


  def validation {
      arg validation_file = 'validation.c';
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
      unrolljam = (['i','j'],[U_I,U_J]),
      regtile = (['i','j'],[RT_I,RT_J])
)
for (i=1; i<=T-1; i++) 
  for (j=1; j<=N-2; j++) 
    a[i][j] = 0.333 * (a[i-1][j-1] + a[i-1][j] + a[i-1][j+1]);

) @*/

/*@ end @*/
/*@ end @*/





