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

    param T1_I[] = [1,16,32,64,128,256,512];
    param T1_J[] = [1,16,32,64,128,256,512];
    param T1_Ia[] = [1,64,128,256,512,1024,2048];
    param T1_Ja[] = [1,64,128,256,512,1024,2048];

    param T2_I[] = [1,16,32,64,128,256,512];
    param T2_J[] = [1,16,32,64,128,256,512];
    param T2_K[] = [1,16,32,64,128,256,512];
    param T2_Ia[] = [1,64,128,256,512,1024,2048];
    param T2_Ja[] = [1,64,128,256,512,1024,2048];
    param T2_Ka[] = [1,64,128,256,512,1024,2048];


    param U1_I[] = range(1,31);
    param U1_J[] = range(1,31);

    param U2_I[] = range(1,31);
    param U2_J[] = range(1,31);
    param U2_K[] = range(1,31);

    
    param RT1_I[] = [1,8,32];
    param RT1_J[] = [1,8,32];

    param RT2_I[] = [1,8,32];
    param RT2_J[] = [1,8,32];
    param RT2_K[] = [1,8,32];



    param SCREP[] = [False,True];
    param VEC1[] = [False,True];
    param OMP1[] = [False,True];
    param VEC2[] = [False,True];
    param OMP2[] = [False,True];



    constraint tileI1 = ((T1_Ia == 1) or (T1_Ia % T1_I == 0));
    constraint tileJ1 = ((T1_Ja == 1) or (T1_Ja % T1_J == 0));
    constraint reg_capacity_1 = (U1_I*U1_J <= 150);
    constraint unroll_limit_1 = ((U1_I == 1) or (U1_J == 1) );
    
    
    
    constraint tileI2 = ((T2_Ia == 1) or (T2_Ia % T2_I == 0));
    constraint tileJ2 = ((T2_Ja == 1) or (T2_Ja % T2_J == 0));
    constraint tileK2 = ((T2_Ka == 1) or (T2_Ka % T2_K == 0));

    constraint reg_capacity_2 = (U2_I*U2_J + U2_I*U2_K + U2_J*U2_K <= 150);
    constraint unroll_limit_2 = ((U2_I == 1) or (U2_J == 1) or (U2_K == 1));

  }
  
  def search
  {
  arg algorithm = 'Randomsearch';
  arg total_runs = 10000;
  }

  def input_params
  {
  param m = 750; 
  param n = 500;
  }

  def input_vars
  {
  decl static double data[m+10][n+10] = random;
  decl static double symmat[m+10][m+10] = random;
  decl static double mean[m+10] = random;
  decl double float_n = 321414134.01;
  decl double eps = 0.005;
  }


) @*/   


int i, j, k;
int ii, jj, kk;
int it, jt, kt;
int iii, jjj, kkk;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(

transform Composite(
    tile = [('j',T1_J,'jj'),('i',T1_I,'ii'),
            (('jj','j'),T1_Ja,'jjj'),(('ii','i'),T1_Ia,'iii')],
    unrolljam = (['j','i'],[U1_J,U1_I]),
    scalarreplace = (SCREP, 'double', 'scv_'),
    regtile = (['j','i'],[RT1_J,RT1_I]),
    vector = (VEC1, ['ivdep','vector always']),
    openmp = (OMP1, 'omp parallel for private(i,j,ii,jj,iii,jjj)')
)
  for (j = 1; j <= m; j++)
    {
      mean[j] = 0.0;
      for (i = 1; i <= n; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }


  for (i = 1; i <= n; i++)
    for (j = 1; j <= m; j++)
      data[i][j] -= mean[j];

transform Composite(
    tile = [('i',T2_I,'ii'),('j',T2_J,'jj'),('k',T2_K,'kk'),
            (('ii','i'),T2_Ia,'iii'),(('jj','j'),T2_Ja,'jjj'),(('kk','k'),T2_Ka,'kkk')],
    unrolljam = (['i','j','k'],[U2_I,U2_J,U2_K]),
    scalarreplace = (SCREP, 'double', 'scv_'),
    regtile = (['i','j','k'],[RT2_I,RT2_J,RT2_K]),
    vector = (VEC2, ['ivdep','vector always']),
    openmp = (OMP2, 'omp parallel for private(iii,jjj,kkk,ii,jj,kk,i,j,k)')
)
  for (k = 1; k <= m-1; k++)
    {
      symmat[k][k] = 1.0;
      for (j = k+1; j <= m; j++)
	{
	  symmat[k][j] = 0.0;
	  for (i = 1; i <= n; i++)
	    symmat[k][j] += (data[i][k] * data[i][j]);
	  symmat[j][k] = symmat[k][j];
	}
    }


) @*/

/*@ end @*/
/*@ end @*/
