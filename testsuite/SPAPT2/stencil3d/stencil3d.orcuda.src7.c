/*@ begin PerfTuning (
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
    param N[] = [200];
    param T[] = [100];
  }

  def input_vars {
    decl static double a[N*N*N] = random;
    decl static double b[N*N*N] = 0;
    decl double f1 = 0.5;
    decl double f2 = 0.6;
  }

) @*/   


#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


int i,j,k,t,n=N;

/*@ begin Loop (

for (t=0; t<=T-1; t++) 
  {

  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
                 cacheBlocks=cache_blocks,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact,
                 streamCount=stream_count)
    for (i=1; i<=n-2; i++)
      for (j=1; j<=n-2; j++)
	for (k=1; k<=n-2; k++) 
	  b[i*n*n + j*n + k] = f1*a[i*n*n + j*n + k] + f2*(a[(i+1)*n*n + j*n +k] 
			+ a[(i-1)*n*n + j*n + k] + a[i*n*n + (j+1)*n + k]
		  	+ a[i*n*n + (j-1)*n + k] + a[i*n*n + j*n + k+1] 
			+ a[i*n*n + j*n + k-1]);

   for (i=1; i<=n-2; i++)
      for (j=1; j<=n-2; j++)
	for (k=1; k<=n-2; k++)
	  a[i*n*n + j*n + k] = b[i*n*n + j*n + k];

  }

) @*/
/*@ end @*/

/*@ end @*/
