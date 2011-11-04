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

    # Cache tiling
    param T1_I[] = [1,16,32,64,128,256,512];
    param T1_J[] = [1,16,32,64,128,256,512];
    param T1_K[] = [1,16,32,64,128,256,512];
    param T1_Ia[] = [1,64,128,256,512,1024,2048];
    param T1_Ja[] = [1,64,128,256,512,1024,2048];
    param T1_Ka[] = [1,64,128,256,512,1024,2048];



    param T2_I[] = [1,16,32,64,128,256,512];
    param T2_J[] = [1,16,32,64,128,256,512];
    param T2_K[] = [1,16,32,64,128,256,512];
    param T2_Ia[] = [1,64,128,256,512,1024,2048];
    param T2_Ja[] = [1,64,128,256,512,1024,2048];
    param T2_Ka[] = [1,64,128,256,512,1024,2048];



    # Unroll-jam 
    param U1_I[]  = range(1,31);
    param U1_J[]  = range(1,31);
    param U1_K[]  = range(1,31);

    param U2_I[]  = range(1,31);
    param U2_J[]  = range(1,31);
    param U2_K[]  = range(1,31);



    # Register tiling
    param RT1_I[] = [1,8,32];
    param RT1_J[] = [1,8,32];
    param RT1_K[] = [1,8,32];

    param RT2_I[] = [1,8,32];
    param RT2_J[] = [1,8,32];
    param RT2_K[] = [1,8,32];




    # Scalar replacement
    param SCR[]  = [False,True];

    # Vectorization
    param VEC1[] = [False,True];
    param VEC2[] = [False,True];

    # Parallelization
    param OMP1[] = [False,True];
    param OMP2[] = [False,True];

    # Constraints
    constraint tileI1 = ((T1_Ia == 1) or (T1_Ia % T1_I == 0));
    constraint tileJ1 = ((T1_Ja == 1) or (T1_Ja % T1_J == 0));
    constraint tileK1 = ((T1_Ka == 1) or (T1_Ka % T1_K == 0));
    constraint reg_capacity1 = (RT1_I*RT1_J + RT1_I*RT1_K + RT1_J*RT1_K <= 150);
    constraint unroll_limit1 = ((U1_I == 1) or (U1_J == 1) or (U1_K == 1));


    constraint tileI2 = ((T2_Ia == 1) or (T2_Ia % T2_I == 0));
    constraint tileJ2 = ((T2_Ja == 1) or (T2_Ja % T2_J == 0));
    constraint tileK2 = ((T2_Ka == 1) or (T2_Ka % T2_K == 0));
    constraint reg_capacity2 = (RT2_I*RT2_J + RT2_I*RT2_K + RT2_J*RT2_K <= 150);
    constraint unroll_limit2 = ((U2_I == 1) or (U2_J == 1) or (U2_K == 1));

  }
  
  def search
  {
   arg algorithm = 'Randomsearch';
   arg total_runs = 10000;
  }

  def input_params
  {
  param N=150;
  param T=50;
  }

  def input_vars
  {
  decl static double a[N][N][N] = random;
  decl static double b[N][N][N] = 0;
  decl double f1 = 0.5;
  decl double f2 = 0.6;
  }

) @*/   


#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


int i,j,k,t;
int it, jt, kt;
int ii, jj, kk;
int iii, jjj, kkk;


/*@ begin Loop (


for (t=0; t<=T-1; t++) 
  {

transform Composite(
    tile = [('i',T1_I,'ii'),('j',T1_J,'jj'),('k',T1_K,'kk'),
            (('ii','i'),T1_Ia,'iii'),(('jj','j'),T1_Ja,'jjj'),(('kk','k'),T1_Ka,'kkk')],
    unrolljam = (['t','i','j'],[U1_I,U1_J,U1_K]),
    scalarreplace = (SCR, 'double'),
    regtile = (['i','j','k'],[RT1_I,RT1_J,RT1_K]),
    vector = (VEC1, ['ivdep','vector always']),
    openmp = (OMP1, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
)
    for (i=1; i<=N-2; i++)
      for (j=1; j<=N-2; j++)
	for (k=1; k<=N-2; k++)
	  b[i][j][k] = f1*a[i][j][k] + f2*(a[i+1][j][k] + a[i-1][j][k] + a[i][j+1][k]
		   + a[i][j-1][k] + a[i][j][k+1] + a[i][j][k-1]);

transform Composite(
    tile = [('i',T2_I,'ii'),('j',T2_J,'jj'),('k',T2_K,'kk'),
            (('ii','i'),T2_Ia,'iii'),(('jj','j'),T2_Ja,'jjj'),(('kk','k'),T2_Ka,'kkk')],
    unrolljam = (['t','i','j'],[U2_I,U2_J,U2_K]),
    scalarreplace = (SCR, 'double'),
    regtile = (['i','j','k'],[RT2_I,RT2_J,RT2_K]),
    vector = (VEC2, ['ivdep','vector always']),
    openmp = (OMP2, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
)
   for (i=1; i<=N-2; i++)
      for (j=1; j<=N-2; j++)
	for (k=1; k<=N-2; k++)
	  a[i][j][k] = b[i][j][k];

  }

) @*/
/*@ end @*/

/*@ end @*/
