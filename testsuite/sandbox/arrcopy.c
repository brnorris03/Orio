/*@ begin PerfTuning (  
  def build
  {
  arg build_command = 'gcc -g -O2 -fopenmp -DDYNAMIC'; 
  arg libs = '-lm '; #-lrt';
  }
  
  def performance_counter  
  { 
    arg repetitions = 10;
  }

  
  def performance_params 
  {
    let BSIZE = 512*32;
    # Cache tiling
    param T_I[] = [1,16,32];
    param T_J[] = [1,16,32,64];
    param T_K[] = [1,16,32,64,128];

    # Array copy
    param ACOPY_A[] = [False,True];
    param ACOPY_B[] = [False,True];
    param ACOPY_C[] = [False,True];
  }

  def input_params 
  {
    param SM[] = [2000];
    param SN[] = [2000];
    param SK[] = [2000];
  }

  def input_vars
  {
    decl int M = SM;
    decl int N = SN;
    decl int K = SK;
    decl dynamic double A[SM][SK] = random;
    decl dynamic double B[SK][SN] = random;
    decl dynamic double C[SM][SN] = 0;
  }

  def search 
  { 
    arg algorithm = 'Randomsearch'; 
    arg total_runs = 10;
  } 

) @*/

void mxm(int M, int N, int K, double **A, double **B, double **C) {
  register int i, ii, j, jj, k, kk;

  /*@ begin Loop(
        transform Composite(
            tile = [('i',T_I,'ii'),('j',T_J,'jj'),('k',T_K,'kk')],
            arrcopy = [(ACOPY_C, 'C[i][j]', [(T_I if T_I>1 else 5000),(T_J if T_J>1 else 5000)], '_copy'),
                       (ACOPY_A, 'A[i][k]', [(T_I if T_I>1 else 5000),(T_K if T_K>1 else 5000)], '_copy'),
                       (ACOPY_B, 'B[k][j]', [(T_K if T_K>1 else 5000),(T_J if T_J>1 else 5000)], '_copy')]
        )

  for (ii=0; ii<=M-1; ii++)
    for (jj=0; jj<=N-1; jj++)
      for (kk=0; kk<=K-1; kk++)
        for (i=ii; i<=min(M-1,ii+31); i++)
          for (j=jj; j<=min(N-1,jj+31); j++)
            for (k=kk; k<=min(K-1,kk+31); k++)
              C[i][j]+=A[i][k]*B[k][j];

  ) @*/

  for (ii=0; ii<=M-1; ii++)
    for (jj=0; jj<=N-1; jj++)
      for (kk=0; kk<=K-1; kk++)
        for (i=ii; i<=min(M-1,ii+31); i++)
          for (j=jj; j<=min(N-1,jj+31); j++)
            for (k=kk; k<=min(K-1,kk+31); k++)
              C[i][j]+=A[i][k]*B[k][j];

  /*@ end @*/
}
/*@ end @*/
