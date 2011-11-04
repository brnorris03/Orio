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








    






}
  
  def search
  {
  arg algorithm = 'Randomsearch';
  arg total_runs = 10000;
  }

  def input_params
  {
  param m = 1000; 
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
    regtile = (['i','j'],[RT2_I,RT2_J]),
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
    regtile = (['i','j'],[RT3_I,RT3_J]),
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
