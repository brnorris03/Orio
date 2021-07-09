void adi(double* X, double* A, double* B) {
/*@ begin PerfTuning (
    def performance_params {
        param thread_count[]  = range(32,1025,32);	# threads per block
        param block_count[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
        param inner_loop_unroll_fact[] = range(1, 33);
	param cache_blocks[] = [False, True];
        param preferred_L1_cache[]  = [16,32,48];
        param stream_count[] = range(1,33);
        param CFLAGS[] = ['-O3','-use_fast_math'];
    }
    def build {
      arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
    }

    def performance_counter {
        arg repetitions = 3;
    }

    def search {
        arg algorithm = 'Randomlocal';
        arg total_runs = 10000;
    }

    def input_params {
        param T[] = [100];
        param N[] = [1024]; 
    }
  
    def input_vars {
        decl static double X[N * (N+20)] = random;
        decl static double A[N * (N+20)] = random;
        decl static double B[N * (N+20)] = random;
    }
) @*/   


#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


int t, n = N;


/*@ begin Loop (
 

for (t=0; t<=T-1; t++) {
  
  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
                 cacheBlocks=cache_blocks,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact,
                 streamCount=stream_count)
  for (i1=0; i1<=n-1; i1++)
    for (i2=1; i2<=n-1; i2++) {
      X[i1 * n + i2] = X[i1 * n + i2] - X[i1 * n + (i2-1)] * A[i1 * n + i2] / B[i1 * n + (i2-1)];
      B[i1 * n + i2] = B[i1 * n + i2] - A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + (i2-1)];
    }

  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact,
                 streamCount=stream_count)
   for (i1=1; i1<=n-1; i1++)
      for (i2=0; i2<=n-1; i2++) {
        X[i1 * n + i2] = X[i1 * n + i2] - X[(i1-1)  * n + i2] * A[i1 * n + i2] / B[(i1-1) * n + i2];
        B[i1 * n + i2] = B[i1 * n + i2] - A[i1 * n + i2] * A[i1 * n + i2] / B[(i1-1) * n + i2];
      }
}

) @*/


/*@ end @*/
/*@ end @*/
}

