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
  #  param PERM[] = [
  #   ['i','j'],
  #   ['j','i'],
  #  ];

    param T2_I[] = [1,16,32,64,128,256,512];
    param T2_J[] = [1,16,32,64,128,256,512];
    param T2_Ia[] = [1,64,128,256,512,1024,2048];
    param T2_Ja[] = [1,64,128,256,512,1024,2048];

    param U2_I[]  = range(1,31); 
    param U2_J[]  = range(1,31); 

    param RT2_I[] = [1,8,32];
    param RT2_J[] = [1,8,32];

    param SCR[]  = [False,True];
    param VEC2[] = [False,True];
    param OMP[] = [False,True];


        
    param U1_J[]  = range(1,31); 
    param VEC1[] = [False,True];
    

    
    constraint tileI2 = ((T2_Ia == 1) or (T2_Ia % T2_I == 0));
    constraint tileJ2 = ((T2_Ja == 1) or (T2_Ja % T2_J == 0));
    constraint reg_capacity = (RT2_I*RT2_J <= 150);
    constraint unroll_limit = ((U2_I == 1) or (U2_J == 1));

  }

  def search 
  { 
    arg algorithm = 'Randomsearch'; 
    arg total_runs = 10000;
  } 
   
  def input_params 
  {
    param N[] = [2000];
  }

  def input_vars
  {
    arg decl_file = 'decl_code.h';
    arg init_file = 'init_code.c';
 }
) @*/ 


int i,j, k,t;
int it, jt, kt;
int ii, jj, kk;
int iii, jjj, kkk;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))



/*@ begin Loop (

for (k=0; k<=N-1; k++) {
   transform Composite(
      unrolljam = (['j'],[U1_J]),
      scalarreplace = (SCR, 'double'),
      vector = (VEC1, ['ivdep','vector always'])
     )
    for (j=k+1; j<=N-1; j++)
      A[k][j] = A[k][j]/A[k][k];

   transform Composite(
      tile = [('i',T2_I,'ii'),('j',T2_J,'jj'),
             (('ii','i'),T2_Ia,'iii'),(('jj','j'),T2_Ja,'jjj')],
      unrolljam = (['i','j'],[U2_I,U2_J]),
      scalarreplace = (SCR, 'double'),
      regtile = (['i','j'],[RT2_I,RT2_J]),
      vector = (VEC2, ['ivdep','vector always']),
      openmp = (OMP, 'omp parallel for private(iii,jjj,ii,jj,i,j)')
    )
    for(i=k+1; i<=N-1; i++)
      for (j=k+1; j<=N-1; j++)
        A[i][j] = A[i][j] - A[i][k]*A[k][j];
  }
) @*/
/*@ end @*/

/*@ end @*/

