/*@ begin PerfTuning (          
  def build  
  {  
  arg build_command = 'gcc -O3 -fopenmp -DDYNAMIC'; 
  arg libs = '-lm -lrt';
  }
     
  def performance_counter           
  {  
    arg repetitions = 35;  
  } 
 
  def performance_params  
  { 


    # Cache tiling
    param T1_T[] = [1,16,32,64,128,256,512];
    param T1_I[] = [1,16,32,64,128,256,512];
    param T1_J[] = [1,16,32,64,128,256,512];

    param T1_Ta[] = [1,64,128,256,512,1024,2048];
    param T1_Ia[] = [1,64,128,256,512,1024,2048];
    param T1_Ja[] = [1,64,128,256,512,1024,2048];

    # Unroll-jam 
    param U1_T[]  = range(1,31);
    param U1_I[]  = range(1,31);
    param U1_J[]  = range(1,31);

    # Register tiling
    param RT1_I[] = [1,8,32];
    param RT1_J[] = [1,8,32];
    param RT1_T[] = [1,8,32];

    # Scalar replacement

    # Vectorization

    # Parallelization

    # Constraints


  } 
 
  def search  
  {
    arg algorithm = 'Randomsearch';  
    arg total_runs = 10000; 
  }

  def input_params  
  {
    param T[] = [100];
    param N[] = [1000];
  } 

  def input_vars
  {
    decl static double A[N][N+17] = random;
  }
) @*/  


#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

int i,j, t;
int it, jt;
int iii, jjj, ttt;
int ii, jj,tt;

/*@ begin Loop(

transform Composite(
    tile = [('t',T1_T,'tt'),('i',T1_I,'ii'),('j',T1_J,'jj'),
            (('tt','t'),T1_Ta,'ttt'),(('ii','i'),T1_Ia,'iii'),(('jj','j'),T1_Ja,'jjj')],
    unrolljam = (['t','i','j'],[U1_T,U1_I,U1_J]),
    regtile = (['t','i','j'],[RT1_T,RT1_I,RT1_J]),
)

for (t=0; t<=T-1; t++)
  for (i=1; i<=N-2; i++)
    for (j=1; j<=N-2; j++)
      A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		 + A[i][j-1] + A[i][j] + A[i][j+1]
		 + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;
) @*/

/*@ end @*/ 
/*@ end @*/ 

