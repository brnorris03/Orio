void MatMatMult(double* A, double* B, double* C, int m, int n, int p) {

/*@ begin PerfTuning (
    def performance_params {
        param thread_count[]  = [32, 64];
        param block_count[]  = range(14,28,14);
        param inner_loop_unroll_fact[] = range(1, 5);
        param preferred_L1_cache[]  = [16, 48];
        param stream_count[] = [1, 2, 4, 8];
        param CFLAGS[] = ['-O3'];
    }

    def input_params
    {
      param N[] = [500];
    }

    def input_vars
    {
        decl static double L[N * N];
        decl static double U[N * N];
        decl static double A[N * (N+13)];
        arg init_file = 'init_code.c';
    }

    def search
    {
        arg algorithm = 'Randomlocal';
        arg total_runs = 1000;
    }

    def build {
      arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
    }

    def performance_counter {
      arg method = 'basic timer';
      arg repetitions = 1;
    }
) @*/

int m = M, p = M, n = M;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(
  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact,
                 streamCount=stream_count)
    for (k=0; k<=N-1; k++) {
      for (j=k+1; j<=N-1; j++)
        A[k][j] = A[k][j]/A[k][k];
  
  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact,
                 streamCount=stream_count)
    for(i=k+1; i<=N-1; i++)
      for (j=k+1; j<=N-1; j++)
        A[i][j] = A[i][j] - A[i][k]*A[k][j];
  }
) @*/

/*@ end @*/
/*@ end @*/


}



