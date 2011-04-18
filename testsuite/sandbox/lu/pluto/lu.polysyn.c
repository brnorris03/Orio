/*@ begin PerfTuning (         
  def build 
  { 
    arg build_command = 'icc -O3 -openmp -I/usr/local/icc/include -lm'; 
  } 
    
  def performance_counter          
  { 
    arg repetitions = 1; 
  }

  def performance_params 
  {
#    param T1_1[] = [1,16,32,64,128];
#    param T1_2[] = [1,16,32,64,128];
#    param T1_3[] = [1,16,32,64,128];
#    param T2_1[] = [1,4,8,16,32];
#    param T2_2[] = [1,4,8,16,32];
#    param T2_3[] = [1,4,8,16,32];

    param T1_1[] = [64];
    param T1_2[] = [256];
    param T1_3[] = [64];
    param T2_1[] = [1];
    param T2_2[] = [1];
    param T2_3[] = [1];

    constraint c1 = (T1_1*T2_1<=1024 and T1_1*T2_1<=1024 and T1_1*T2_1<=1024);
    constraint c2 = ((T1_1 == T1_3) and (T2_1 == T2_3));

    param U1[] = [1];
    param U2[] = [1];
    param U3[] = [7];

    constraint c3 = (U1*U2*U3<=512);

    param PERM[] = [
      #[0,1,2],
      #[0,2,1],
      #[1,0,2],
      #[1,2,0],
      [2,0,1],
      #[2,1,0],
      ];

    param PAR[] = [True];
    param SCREP[] = [False];
    param IVEC[] = [True];
  }

  def search 
  { 
    arg algorithm = 'Exhaustive'; 
#    arg algorithm = 'Simplex'; 
#    arg time_limit = 5;
#    arg total_runs = 1;
  } 
   
  def input_params 
  {
    param N[] = [1024];
  }

  def input_vars
  {
    arg decl_file = 'decl_code.h';
    arg init_file = 'init_code.c';
  }
) @*/ 

register int i,j,k;
register int c1t, c2t, c3t, c4t, c5t, c6t, c7t, c8t, c9t, c10t, c11t, c12t;
register int newlb_c1, newlb_c2, newlb_c3, newlb_c4, newlb_c5, newlb_c6,
  newlb_c7, newlb_c8, newlb_c9, newlb_c10, newlb_c11, newlb_c12;
register int newub_c1, newub_c2, newub_c3, newub_c4, newub_c5, newub_c6,
  newub_c7, newub_c8, newub_c9, newub_c10, newub_c11, newub_c12;


/*@ begin PolySyn(    
  parallel = PAR;
  tiles = [T1_1,T1_2,T1_3,T2_1,T2_2,T2_3];
  permut = PERM;
  unroll_factors = [U1,U2,U3];
  scalar_replace = SCREP;
  vectorize = IVEC;
    
  profiling_code = 'lu_profiling.c';
  compile_cmd = 'gcc';
  compile_opts = '-lm';
  ) @*/

/* pluto start (N) */
for (k=0; k<=N-1; k++)
  {
    for (j=k+1; j<=N-1; j++)
      A[k][j] = A[k][j]/A[k][k];
    for(i=k+1; i<=N-1; i++)
      for (j=k+1; j<=N-1; j++)
        A[i][j] = A[i][j] - A[i][k]*A[k][j];
  }
/* pluto end */

/*@ end @*/
/*@ end @*/

