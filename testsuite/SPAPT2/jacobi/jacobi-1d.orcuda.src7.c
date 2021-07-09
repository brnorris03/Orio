/*@ begin PerfTuning (          
  def build  {  
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
 
  def input_params {
    param T[] = [10000];
    param N[] = [10000];
  }

  def input_vars { 
    decl int t = T;
    decl int n = N;
    decl static double a[T * N] = 0;
    arg init_file = 'jacobi-1d_init2_code.c';
  } 

  def search  {
    arg algorithm = 'Randomlocal';  
    arg total_runs = 10000; 
  }

) @*/  


#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(

  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
                 cacheBlocks=cache_blocks,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact,
                 streamCount=stream_count)
  for (i=1; i<=t-1; i++) 
    for (j=1; j<=n-2; j++) 
      a[i*n + j] = 0.333 * (a[(i-1)*n +(j-1)] + a[(i-1)*n + j] + a[(i-1)*n +(j+1)]);

) @*/

/*@ end @*/
/*@ end @*/




