
//multiply matrices together
//pre all matrices are initialized, c shouldn't have any important data in it
//     mat b is trnasposed
//     rows in b == cols in a
//     c is initialized to the same size as b
//post mat_c has the result of multipling mat_a and mat_b
void multiply_matrix_t(double** restrict mat_a, int rows_a, int cols_a, 
                       double** restrict mat_b, int cols_b, 
                       double** restrict mat_c) {


  /*@ begin PerfTuning (
    def build {
      arg build_command = 'gcc-9 -fopenmp -mcmodel=large @CFLAGS';
    } 
    def performance_counter {
      arg repetitions = 5;
    }
    def performance_params {  

      param CFLAGS[] = ['-O3', '-O2 -march=skylake-avx512', '-O3 -march=skylake-avx512', '-O3 -march=skylake', '-Ofast -march=skylake-avx512'];

      param T1_J[] = [1,64,128,256,512];
      param T1_K[] = [1,64,128,256,512];
      param T1_Ja[] = [1,64,128,256,512,1024,2048];
      param T1_Ka[] = [1,64,128,256,512,1024,2048];


      param U_J[] = [1] + list(range(2,17,2));
      param U_K[] = [1] + list(range(2,17,2));

      param SCREP[] = [False,True];

      param VEC[] = [False,True];

      param OMP = True;
      
      constraint tileJ = (((T1_Ja == 1) or (T1_Ja % T1_J == 0)) and ((T1_Ja == 1) or (T1_Ja > T1_J)));
      constraint tileK = (((T1_Ka == 1) or (T1_Ka % T1_K == 0)) and ((T1_Ka == 1) or (T1_Ka > T1_K)));
      constraint reg_capacity = (U_J*U_K <= 150);
      constraint unroll_limit = ((U_J == 1) or (U_K == 1));
    }
    def input_params {
      let N = [4096,8192,16384];
      param rows_a[] = N;
      param cols_a[] = N;
      param cols_b[] = N;

      #constraint square = ((rows_a == cols_a) and (rows_a == cols_b));
    }
    def input_vars {
      decl static double A[rows_a][cols_a] = random;
      decl static double B[cols_b][cols_a] = random;
      decl static double C[rows_a][cols_b] = random;
    }
    def search {
      arg algorithm = 'Mlsearch';
      arg total_runs = 20;
    }
  ) @*/

  int i, j, k;
  int ii, jj;
  int iii, jjj;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

// b is transposed

/*@ begin Loop(
  transform Composite(
    tile = [('j',T1_J,'jj'),('k',T1_K,'kk'),
            (('jj','j'),T1_Ja,'jjj'),(('kk','k'),T1_Ka,'kkk')],
    unrolljam = (['j','k'],[U_J,U_K]),
    scalarreplace = (SCREP, 'double', 'scv_'),
    vector = (VEC, ['ivdep','vector always']),
    openmp = (OMP, 'omp parallel for private(iii,jjj,ii,jj,i,j,k)')
  )
  for(i=0; i<=rows_a-1; i++) 
    for(j=0; j<=cols_b-1; j++)   
      for(k=0; k<=cols_a-1; k++) 
        C[i][j] = C[i][j] + A[i][k] * B[j][k]; 
) @*/

/*@ end @*/
/*@ end @*/


}
