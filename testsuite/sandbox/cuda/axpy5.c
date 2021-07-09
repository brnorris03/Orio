void axpy5(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4, double a5, double *x5) {
    
  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param TC[] = [32,64];    		# threads per block
          param BC[] = range(108,1081,108);	# number of blocks, grid.x; multiples of SMs, 108 for A100
          param SC[] = range(1,31); 		# number of streams
          param CB[] = [False, True];		# cache blocks
          param PL[] = [16,32,48];		# prefered L1 cache (KB)
          param CFLAGS[] = ['-O0', '-O3', '-use_fast_math']; # -O3 is default
        }
        def build {
          arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
        }
        def input_params {
          param N[] = [1000];
        }
        def input_vars {
          decl static double y[N] = 0;
          decl double a1 = random;
          decl double a2 = random;
          decl double a3 = random;
          decl double a4 = random;
          decl double a5 = random;
          decl static double x1[N] = random;
          decl static double x2[N] = random;
          decl static double x3[N] = random;
          decl static double x4[N] = random;
          decl static double x5[N] = random;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 3;
        }
	def search {
	  arg algorithm = 'Randomlocal';
	  arg total_runs = 100; 
	}
  ) @*/

  int n=N;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, streamCount=SC, cacheBlocks=CB, preferL1Size=PL)

  for (i=0; i<=n-1; i++)
    y[i]+=a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    y[i]+=a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];

  /*@ end @*/
  /*@ end @*/
}
