void MatMatMult(double* A, double* B, double* C, int n) {

/*@ begin PerfTuning (
    def performance_params {
        param thread_count[]  = range(32,1025,32);	# threads per block
        param block_count[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
        param inner_loop_unroll_fact[] = range(1,33);
	    param cache_blocks[] = [False, True];
        param preferred_L1_cache[]  = [16,32,48];
        param stream_count[] = range(1,31);
        param CFLAGS[] = ['-O3','-use_fast_math'];
    }

    def input_params {
        param M = 2048;
    }

    def input_vars {
        decl static double A[M * M] = random;
        decl static double B[M * M] = random;
        decl static double C[M * M] = 0;
    }

    def search {
        arg algorithm = 'Mlsearch';
        arg total_runs = 100;
    }

    def build {
      arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
    }

    def performance_counter {
      arg method = 'basic timer';
      arg repetitions = 3;
    }
) @*/

int n = M;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(
  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
		         cacheBlocks=cache_blocks,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact,
                 streamCount=stream_count)
  for(i=0; i<=n-1; i++)
    for(j=0; j<=n-1; j++) {
      for(k=0; k<=n-1; k++){
        C[i*n+j] += A[i*n+k]*B[k*n+j];
      }
    }
) @*/

/*@ end @*/
/*@ end @*/


}



