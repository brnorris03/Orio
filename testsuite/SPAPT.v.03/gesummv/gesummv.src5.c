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
    param T1_I[] = [1,16,32,64,128,256,512];
    param T1_J[] = [1,16,32,64,128,256,512];
    param T1_Ia[] = [1,64,128,256,512,1024,2048];
    param T1_Ja[] = [1,64,128,256,512,1024,2048];


    # Unroll-jam 
    param U1_I[]  = range(1,31);
    param U1_J[]  = range(1,31);


    # Register tiling
    param RT1_I[] = [1,8,32];
    param RT1_J[] = [1,8,32];


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
  param N[] = [20000];
  }
  
  def input_vars
  {
  arg decl_file = 'decl.h';
  arg init_file = 'init.c';
  }
) @*/


int i,j,t;
int it, jt, kt;
int ii, jj ;
int iii, jjj;
double* tmp=(double*)malloc(n*sizeof(double));

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


/*@ begin Loop (

  transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),
            (('ii','i'),T1_Ia,'iii'),(('jj','j'),T1_Ja,'jjj')],
    unrolljam = (['i','j'],[U1_I,U1_J]),
    regtile = (['i','j'],[RT1_I,RT1_J])
)
  for (i=0; i<=n-1; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (j=0; j<=n-1; j++) {
      tmp[i] = A[i*n+j]*x[j] + tmp[i];
      y[i] = B[i*n+j]*x[j] + y[i];
    }
    y[i] = a*tmp[i] + b*y[i];
  }

) @*/

/*@ end @*/

/*@ end @*/




  


