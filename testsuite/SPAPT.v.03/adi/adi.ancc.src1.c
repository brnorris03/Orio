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
    param SCR1[]  = [False,True];
    param SCR2[]  = [False,True];

    # Vectorization
    param VEC1[] = [False,True];
    param VEC2[] = [False,True];

    # Parallelization
    #param OMP[] = [False,True];

    # Constraints
    constraint tileT1I1 = ((T1_I1a == 1) or (T1_I1a % T1_I1 == 0));
    constraint tileT1I2 = ((T1_I2a == 1) or (T1_I2a % T1_I2 == 0));
    constraint tileT2I1 = ((T2_I1a == 1) or (T2_I1a % T2_I1 == 0));
    constraint tileT2I2 = ((T2_I2a == 1) or (T2_I2a % T2_I2 == 0));
    constraint reg_capacity_1 = (RT1_I1*RT1_I2 <= 150);
    constraint reg_capacity_2 = (RT2_I1*RT2_I2 <= 150);
    constraint unroll_limit_1 =  (U1_I1 == 1) or (U1_I2 == 1); 
    constraint unroll_limit_2 =  (U2_I1 == 1) or (U2_I2 == 1) ;

  }
  
  def search
  {
  arg algorithm = 'Randomsearch';
  arg total_runs = 10000;
  }

  def input_params
  {
  param T[] = [64];
  param N[] = [512]; 
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
    scalarreplace = (SCR1, 'double'),
    regtile = (['i1','i2'],[RT1_I1,RT1_I2]),
    vector = (VEC1, ['ivdep','vector always'])
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
    scalarreplace = (SCR2, 'double'),
    regtile = (['i1','i2'],[RT2_I1,RT2_I2]),
    vector = (VEC2, ['ivdep','vector always'])
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


