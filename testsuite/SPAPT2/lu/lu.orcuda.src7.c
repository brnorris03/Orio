/*@ begin PerfTuning (         
  def build { 
    arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
  } 
    
  def performance_counter { 
    arg repetitions = 3; 
  }

  def performance_params {
    param thread_count[]  = range(32,1025,32);	# threads per block
    param block_count[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
    param inner_loop_unroll_fact[] = range(1, 33);
    param cache_blocks[] = [False, True];
    param preferred_L1_cache[]  = [16,32,48];
    param stream_count[] = range(1,33);
    param CFLAGS[] = ['-O3','-use_fast_math'];
  }

  def search { 
    arg algorithm = 'Randomlocal'; 
    arg total_runs = 10000;
  } 
   
  def input_params {
    param N[] = [2000];
  }

  def input_vars {
    decl int n = N;
    decl int k = 0;
    decl static double L[N*N] = 0;
    decl static double U[N*N] = 0;
    decl static double A[N*N] = 0; 
    arg init_file = 'init_code2.c';
 }

) @*/ 



#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop (

  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
                 cacheBlocks=cache_blocks,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact,
                 streamCount=stream_count)
  for (k=0; k<=n-1; k++) {
    for(j=k+1; j<=n-1; j++) {
      A[k*n+i] = A[k*n+i]/A[k*n+i];
    }
    for(i=k+1; i<=n-1; i++) {
      for (j=k+1; j<=n-1; j++)
        A[i*n+j] = A[i*n+j] - A[i*n+k]*A[k*n+j];
    }
  }
) @*/
/*@ end @*/

/*@ end @*/

