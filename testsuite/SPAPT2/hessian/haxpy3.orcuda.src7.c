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
    param SIZE = 10000;
    param N = 2000;
  }            

  def input_vars
  { 
    decl int n = N;
    decl static double X0[N*N] = random;
    decl static double X1[N*N] = random;
    decl static double X2[N*N] = random;
    decl static double Y[N*N] = 0;
    decl static double u0[N] = random;
    decl static double u1[N] = random;
    decl static double u2[N] = random;
    decl double a0 = 32.12;
    decl double a1 = 3322.12;
    decl double a2 = 1.123;
    decl double b00 = 1321.9;
    decl double b01 = 21.55;
    decl double b02 = 10.3;
    decl double b11 = 1210.313;
    decl double b12 = 9.373;
    decl double b22 = 1992.31221;
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
  for (i=0; i<=n-1; i++)
    for (j=0; j<=n-1; j++) {
      Y[i*n+j]=a0*X0[i*n+j] + a1*X1[i*n+j] + a2*X2[i*n+j]
	+ 2.0*b00*u0[i]*u0[j]
	+ 2.0*b11*u1[i]*u1[j]
	+ 2.0*b22*u2[i]*u2[j]
	+ b01*u0[i]*u1[j] + b01*u1[i]*u0[j] 
	+ b02*u0[i]*u2[j] + b02*u2[i]*u0[j]
	+ b12*u1[i]*u2[j] + b12*u2[i]*u1[j];
    }

) @*/

/*@ end @*/
/*@ end @*/


