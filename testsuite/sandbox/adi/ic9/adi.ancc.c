/*@ begin PerfTuning (
  def build
  {
  arg build_command = 'icc -fast -openmp -lm -lrt';
  }

  def performance_counter
  {
  arg method = 'basic timer';
  arg repetitions = 35;
  }  
  
  def performance_params
  {  
    param T1_1[] = [1,4,8,16,32,64,128,256,512];
    param T1_2[] = [1,4,8,16,32,64,128,256,512];
    param T1_3[] = [1,4,8,16,32,64,128,256,512];
    param T2_1[] = [1,4,8,16,32,64,128,256,512];
    param T2_2[] = [1,4,8,16,32,64,128,256,512];

    param UF1[] = range(1,31);
    param UF2[] = range(1,31);
    param UF3[] = range(1,31);
    param UF4[] = range(1,31);
    param UF5[] = range(1,31);


    param PERM1[] = [
      ['i1','i2'],
      ['i2','i1'],
    ];

    param PERM2[] = [
      ['i1','i2'],
      ['i2','i1'],
    ];

    param PAR1[] = [False,True];
    param PAR2[] = [False,True];
    param PAR3[] = [False,True];
    param PAR4[] = [False,True];
    param PAR5[] = [False,True];
    param SCREP[] = [False,True];
    param VEC1[] = [False,True];
    param VEC2[] = [False,True];
    param VEC3[] = [False,True];
    param VEC4[] = [False,True];
    param VEC5[] = [False,True];

  }
  
  def search
  {
  arg algorithm = 'Randomsearch';
  arg total_runs = 10;
  }

  def input_params
  {
  param T[] = [10];
  param N[] = [100]; 
  }
  
  def input_vars {
  decl static double X[N][N+20] = random;
  decl static double A[N][N+20] = random;
  decl static double B[N][N+20] = random;
  }
) @*/   

register int i1,i2,t;  

/*@ begin Loop (
 
transform Composite(
scalarreplace = (SCREP, 'double'),
vector = (VEC1, ['ivdep', 'vector always'])
)
transform UnrollJam(ufactor=UF1,parallelize=PAR1)
for (t=0; t<=T-1; t++) 
  {
  transform Composite(
   vector = (VEC2, ['ivdep', 'vector always'])
  )
  transform UnrollJam(ufactor=UF2,parallelize=PAR2)
  for (i1=0; i1<=N-1; i1++) 
    transform Composite(
    vector = (VEC3, ['ivdep', 'vector always'])
    )
    transform UnrollJam(ufactor=UF3,parallelize=PAR3)
    for (i2=1; i2<=N-1; i2++) 
    {
     X[i1][i2] = X[i1][i2] - X[i1][i2-1] * A[i1][i2] / B[i1][i2-1];
     B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2-1];
     }
       
   transform Composite(
   permut = [PERM2],
   scalarreplace = (SCREP, 'double'),
   regtile = (['i1','i2'],[T2_1,T2_2]),
   vector = (VEC4, ['ivdep', 'vector always'])
   )
   transform UnrollJam(ufactor=UF4,parallelize=PAR4)
   for (i1=1; i1<=N-1; i1++) 
     transform Composite(
     vector = (VEC5, ['ivdep', 'vector always'])
      )
      transform UnrollJam(ufactor=UF5,parallelize=PAR5) 
      for (i2=0; i2<=N-1; i2++) 
      {
      X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2];
      B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2];
       }
  }

) @*/


/*@ end @*/
/*@ end @*/


