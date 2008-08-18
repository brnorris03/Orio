/*@ begin PerfTuning (          
  def build  
  {  
    arg build_command = 'icc -O3 -openmp -lm';  
  }
     
  def performance_counter           
  {  
    arg repetitions = 1;  
  } 
 
  def performance_params  
  { 
# [1,8,16,32,64,128,256]; 
# [1,4,8,16,32];

    param T1_1[] = [16];
    param T1_2[] = [8];
    param T1_3[] = [512];
    param T2_1[] = [1];
    param T2_2[] = [1];
    param T2_3[] = [1];
 
    constraint c1 = (T1_1*T2_1<=1024 and T1_1*T2_1<=1024 and T1_1*T2_1<=1024);

    param U1[] = [3];
    param U2[] = [1];
    param U3[] = [4];

    constraint c2 = (U1*U2*U3<=256);

    param PERM[] = [
     [0,1,2],
#     [0,2,1],
#     [1,0,2],
#     [1,2,0],
#     [2,0,1],
#     [2,1,0],
    ];

    param PAR[] = [False];
    param SCREP[] = [True];
    param IVEC[] = [True];
    param RECTILE[] = [True];
  } 
 
  def search  
  {
    arg algorithm = 'Exhaustive';  
#    arg algorithm = 'Simplex';  
#    arg total_runs = 5; 
  }

  def input_params  
  {
    param T[] = [500];
    param N[] = [4000];
  } 

  def input_vars
  {
    decl static double A[N][N+17] = random;
  }
) @*/  

register int i,j,k,t; 
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
  rect_regtile = RECTILE;
  scalar_replace = SCREP;
  vectorize = IVEC;

  profiling_code = 'seidel_profiling.c';
  compile_cmd = 'gcc';
  compile_opts = '-lm';
) @*/

/* pluto start (T,N) */
for (t=0; t<=T-1; t++)
  for (i=1; i<=N-2; i++)
    for (j=1; j<=N-2; j++)
      A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		 + A[i][j-1] + A[i][j] + A[i][j+1]
		 + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;
/* pluto end */

/*@ end @*/ 
/*@ end @*/ 

