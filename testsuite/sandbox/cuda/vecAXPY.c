void VecAXPY(int n, double a, double *x, double *y) {

  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param TC[] = [32,64];    		# threads per block
          param BC[] = range(108,1081,108);	# number of blocks, grid.x
          param SC[] = range(1,31); 		# number of streams
          param CB[] = [False, True];		# cache blocks
          param PL[] = [16,32,48];		# prefered L1 cache (KB)
          param CFLAGS[] = ['-O0', '-O3', '-use_fast_math'];
        }
        def build {
          arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
        }
        def input_params {
          param N[] = [1000];
        }
        def input_vars {
          decl double a = random;
          decl static double x[N] = random;
          decl static double y[N] = 0;
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

  /*@ begin Loop(
      transform CUDA(threadCount=TC, blockCount=BC, streamCount=SC, cacheBlocks=CB, 
		     preferL1Size=PL)

      for (i=0; i<=n-1; i++)
        y[i]+=a*x[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    y[i]+=a*x[i];

  /*@ end @*/
  /*@ end @*/
}
