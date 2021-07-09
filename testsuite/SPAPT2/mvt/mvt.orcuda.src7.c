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
    param inner_loop_unroll_fact[] = range(1,33);
    param cache_blocks[] = [False, True];		
    param preferred_L1_cache[]  = [16,32,48];
    param CFLAGS[] = ['-O3','-use_fast_math'];
  }

  def search {
    arg algorithm = 'Randomlocal';
    arg total_runs = 10000;
  }
   
  def input_params {
    let SIZE = 10000;
    param MSIZE = SIZE;
    param NSIZE = SIZE;
    param M = SIZE;
    param N = SIZE;
  }

  def input_vars {
    decl int m = M;
    decl int n = N;
    decl static double a[M*N] = random;
    decl static double y_1[N] = random;
    decl static double y_2[M] = random;
    decl static double x1[M] = 0;
    decl static double x2[N] = 0;
  }
) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(

  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
                 cacheBlocks=cache_blocks,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact)
  for (i=0;i<=m-1;i++)
    for (j=0;j<=n-1;j++) { 
      x1[i]=x1[i] + a[i*n+j] * y_1[j]; 
      x2[j]=x2[j] + a[i*n+j] * y_2[i]; 
    } 
) @*/


/*@ end @*/
/*@ end @*/


