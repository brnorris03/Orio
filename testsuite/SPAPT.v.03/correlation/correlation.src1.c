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

#    param T1_I[] = [1,16,32,64,128,256,512];
#    param T1_J[] = [1,16,32,64,128,256,512];
#    param T1_Ia[] = [1,64,128,256,512,1024,2048];
#    param T1_Ja[] = [1,64,128,256,512,1024,2048];

    param T2_I[] = [1,16,32,64,128,256,512];
    param T2_J[] = [1,16,32,64,128,256,512];
    param T2_Ia[] = [1,64,128,256,512,1024,2048];
    param T2_Ja[] = [1,64,128,256,512,1024,2048];


    param T3_I[] = [1,16,32,64,128,256,512];
    param T3_J[] = [1,16,32,64,128,256,512];
    param T3_Ia[] = [1,64,128,256,512,1024,2048];
    param T3_Ja[] = [1,64,128,256,512,1024,2048];

#    param T4_I[] = [1,16,32,64,128,256,512];
#    param T4_J[] = [1,16,32,64,128,256,512];
#    param T4_K[] = [1,16,32,64,128,256,512];

#    param T4_Ia[] = [1,64,128,256,512,1024,2048];
#    param T4_Ja[] = [1,64,128,256,512,1024,2048];
#    param T4_Ka[] = [1,64,128,256,512,1024,2048];

#    param U1_I[] = range(1,31);
#    param U1_J[] = range(1,31);

    param U2_I[] = range(1,31);
    param U2_J[] = range(1,31);

    param U3_I[] = range(1,31);
    param U3_J[] = range(1,31);


#    param U4_I[] = range(1,31);
#    param U4_J[] = range(1,31);
#    param U4_K[] = range(1,31);
    
#    param RT1_I[] = [1,8,32];
#    param RT1_J[] = [1,8,32];

    param RT2_I[] = [1,8,32];
    param RT2_J[] = [1,8,32];

    param RT3_I[] = [1,8,32];
    param RT3_J[] = [1,8,32];
    
#    param RT4_I[] = [1,8,32];
#    param RT4_J[] = [1,8,32];
#    param RT4_K[] = [1,8,32];

    param SCREP[] = [False,True];
#    param VEC1[] = [False,True];
#    param OMP1[] = [False,True];
    param VEC2[] = [False,True];
    param OMP2[] = [False,True];
    param VEC3[] = [False,True];
    param OMP3[] = [False];
#    param VEC4[] = [False,True];
#    param OMP4[] = [False,True];


#    constraint tileI1 = ((T1_Ia == 1) or (T1_Ia % T1_I == 0));
#    constraint tileJ1 = ((T1_Ja == 1) or (T1_Ja % T1_J == 0));

    constraint tileI2 = ((T2_Ia == 1) or (T2_Ia % T2_I == 0));
    constraint tileJ2 = ((T2_Ja == 1) or (T2_Ja % T2_J == 0));


    constraint tileI3 = ((T3_Ia == 1) or (T3_Ia % T3_I == 0));
    constraint tileJ3 = ((T3_Ja == 1) or (T3_Ja % T3_J == 0));


#    constraint tileI4 = ((T4_Ia == 1) or (T4_Ia % T4_I == 0));
#    constraint tileJ4 = ((T4_Ja == 1) or (T4_Ja % T4_J == 0));
#    constraint tileK4 = ((T4_Ka == 1) or (T4_Ka % T4_K == 0));
    
#    constraint reg_capacity_1 = (U1_I*U1_J <= 150);
#    constraint unroll_limit_1 = ((U1_I == 1) or (U1_J == 1) );

    constraint reg_capacity_2 = (U2_I*U2_J <= 150);
    constraint unroll_limit_2 = ((U2_I == 1) or (U2_J == 1) );

    constraint reg_capacity_3 = (U3_I*U3_J <= 150);
    constraint unroll_limit_3 = ((U3_I == 1) or (U3_J == 1) );

#    constraint reg_capacity_4 = (U4_I*U4_J + U4_I*U4_K + U4_J*U4_K <= 150);
#    constraint unroll_limit_4 = ((U4_I == 1) or (U4_J == 1) or (U4_K == 1));

}
  
  def search
  {
  arg algorithm = 'Randomsearch';
  arg total_runs = 10000;
  }

  def input_params
  {
  param m = 500; 
  param n = 500;
  }

  def input_vars
  {
  decl static double data[m+10][m+10] = random;
  decl static double symmat[m+10][m+10] = random;
  decl static double stddev[m+10] = random;
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
#define sqrt_of_array_cell(x,j) sqrt(x[j])


/*@ begin Loop(


  for (j = 1; j <= m; j++)
    {
      mean[j] = 0.0;
      for (i = 1; i <= n; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


transform Composite(
    tile = [('i',T2_I,'ii'),('j',T2_J,'jj'),
            (('ii','i'),T2_Ia,'iii'),(('jj','j'),T2_Ja,'jjj')],
    unrolljam = (['i','j'],[U2_I,U2_J]),
    scalarreplace = (SCREP, 'double', 'scv_'),
    regtile = (['i','j'],[RT2_I,RT2_J]),
    vector = (VEC2, ['ivdep','vector always']),
    openmp = (OMP2, 'omp parallel for private(i,j,ii,jj,iii,jjj)')
)
  for (j = 1; j <= m; j++)
    {
      stddev[j] = 0.0;
      for (i = 1; i <= n; i++)
	stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = sqrt_of_array_cell(stddev, j);
      
      stddev[j] = 1.0;
    }


transform Composite(
    tile = [('i',T3_I,'ii'),('j',T3_J,'jj'),
            (('ii','i'),T3_Ia,'iii'),(('jj','j'),T3_Ja,'jjj')],
    unrolljam = (['i','j'],[U3_I,U3_J]),
    scalarreplace = (SCREP, 'double', 'scv_'),
    regtile = (['i','j'],[RT3_I,RT3_J]),
    vector = (VEC3, ['ivdep','vector always']),
    openmp = (OMP3, 'omp parallel for private(i,j,ii,jj,iii,jjj)')
)
  for (i = 1; i <= n; i++)
    for (j = 1; j <= m; j++)
      {
	data[i][j] -= mean[j];
	data[i][j] /= sqrt(float_n) * stddev[j];
      }


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

 symmat[m][m] = 1.0;





) @*/

/*@ end @*/
/*@ end @*/
