
/*@ begin PerfTuning (        
  def build
  {
    arg build_command = 'icc  -DNCONT=10000 -DCONT=1 ';
  }
   
  def performance_counter         
  {
    arg repetitions = 35;
  }
  

 def performance_params
  {
 
    param U_I[] = range(1,31);
    param U_J[] = range(1,31);
    param U_K[] = range(1,31);

   }


  def search
  {
    arg algorithm = 'Randomsearch';
    arg total_runs=50;
  }


  def input_params
  {
    param CONT = 10000;
    param NCONT = 100;
  }  

  def input_vars
  {
    decl int M = NCONT;
    decl int N = NCONT;
    decl int K = CONT;
    decl dynamic double A[M][K] = random;
    decl dynamic double B[K][N] = random;
    decl dynamic double C[M][N] = 0;
   }
            
) @*/

int i, j, k;
int ii, jj, kk;
int iii, jjj, kkk;

/*@ begin Loop(
  
  transform UnrollJam(ufactor=U_I)
  for(i=0; i<=M-1; i++) 
    transform UnrollJam(ufactor=U_J)
    for(j=0; j<=N-1; j++)   
      transform UnrollJam(ufactor=U_K)
      for(k=0; k<=K-1; k++) 
        C[i][j] = C[i][j] + A[i][k] * B[k][j]; 

) @*/

  for(i=0; i<=M-1; i++) 
    for(j=0; j<=N-1; j++)   
      for(k=0; k<=K-1; k++) 
        C[i][j] = C[i][j] + A[i][k] * B[k][j]; 

/*@ end @*/
/*@ end @*/


