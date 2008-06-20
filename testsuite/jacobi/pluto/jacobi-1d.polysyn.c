
/*@ begin PolySyn( 
  parallel = False;
  tiles = [1,1,1,1];
  permut = [0,1];
  unroll_factors = [1,1];
  scalar_replace = False;
  vectorize = False;
 
  profiling_code = 'jacobi-1d_profiling.c'; 
  compile_cmd = 'gcc'; 
  compile_opts = '-lm'; 
) @*/ 

/* pluto start (T,N) */
for (t=1; t<=T-1; t++) 
  for (i=1; i<=N-2; i++) 
    a[t][i] = 0.333 * (a[t-1][i-1] + a[t-1][i] + a[t-1][i+1]);
/* pluto end */

/*@ end @*/

