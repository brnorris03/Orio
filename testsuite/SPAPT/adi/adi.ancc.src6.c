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

    param T1_I1[] = [1,16,32,64,128,256,512];
    param T1_I2[] = [1,16,32,64,128,256,512];
    param T1_I1a[] = [1,64,128,256,512,1024,2048];
    param T1_I2a[] = [1,64,128,256,512,1024,2048];

    param T2_I1[] = [1,16,32,64,128,256,512];
    param T2_I2[] = [1,16,32,64,128,256,512];
    param T2_I1a[] = [1,64,128,256,512,1024,2048];
    param T2_I2a[] = [1,64,128,256,512,1024,2048];


    # Unroll-jam 
    param U1_I1[] = range(1,31); 
    param U1_I2[] = range(1,31); 
    param U2_I1[] = range(1,31); 
    param U2_I2[] = range(1,31); 
    

    # Register tiling
    param RT1_I1[] = [1,8,32];
    param RT1_I2[] = [1,8,32];
    param RT2_I1[] = [1,8,32];
    param RT2_I2[] = [1,8,32];
    

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
  param T[] = [256];
  param N[] = [1024]; 
  }
  
  def input_vars {
  decl static double X[N][N+20] = random;
  decl static double A[N][N+20] = random;
  decl static double B[N][N+20] = random;
  }
) @*/   


#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


int i,j,t;
int i1,i2,i1t,i2t;
int it, jt, kt;
int ii, jj, kk;
int iii, jjj, kkk;



/*@ begin Loop (
 

for (t=0; t<=T-1; t++) 
  {
  
  transform Composite(
    tile = [('i1',T1_I1,'ii'),('i2',T1_I2,'jj'),
            (('ii','i1'),T1_I1a,'iii'),(('jj','i2'),T1_I2a,'jjj')],
    unrolljam = (['i1','i2'],[U1_I1,U1_I2]),
    regtile = (['i1','i2'],[RT1_I1,RT1_I2])
)
  for (i1=0; i1<=N-1; i1++) 
    for (i2=1; i2<=N-1; i2++) 
    {
     X[i1][i2] = X[i1][i2] - X[i1][i2-1] * A[i1][i2] / B[i1][i2-1];
     B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2-1];
     }

  transform Composite(
    tile = [('i1',T2_I1,'ii'),('i2',T2_I2,'jj'),
            (('ii','i1'),T2_I1a,'iii'),(('jj','i2'),T2_I2a,'jjj')],
    unrolljam = (['i1','i2'],[U2_I1,U2_I2]),
    regtile = (['i1','i2'],[RT2_I1,RT2_I2])
)
   for (i1=1; i1<=N-1; i1++) 
      for (i2=0; i2<=N-1; i2++) 
      {
      X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2];
      B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2];
       }
  }


) @*/


/*@ end @*/
/*@ end @*/


