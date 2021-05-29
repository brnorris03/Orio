void BICG(double* A, double* p, double* q, double* r, double* s, int NX, int NY) {

  register int i,j;
  int tmp;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,1025,32);	# threads per block
          param BC[]  = range(108,1081,108);	# number of blocks, grid.x; multiples of # SMs, 108 for A100
          param UIF[] = range(1,31);		# number of streams
          param PL[]  = [16,32,48];		# preferred L1 cache size
          param SC[]  = [1];  			# number of streams
          param CFLAGS[] = ['-O0','-O3','-use_fast_math'];
        }
        def input_params {
          param NX[] = [1024];
          param NY[] = [1024];
          constraint c1 = (NX==NY);
        }
        def input_vars {
          decl static double A[NX*NY] = random;
          decl static double p[NY]    = random;
          decl static double q[NX]    = 0;
          decl static double r[NX]    = random;
          decl static double s[NY]    = 0;
        }
        def build {
          arg build_command = 'nvcc -g -lineinfo -arch=sm_75 @CFLAGS';
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

  int nx=NX;
  int ny=NY;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL, unrollInner=UIF, streamCount=SC)

  for (i = 0; i <= nx-1; i++) {
    q[i] = 0;
    for (j = 0; j <= ny-1; j++) {
      s[j] = s[j] + r[i]*A[i+j];
      q[i] = q[i] + A[i+j]*p[j];
    }
   }

   ) @*/

  for (i = 0; i <= nx-1; i++) {
    q[i] = 0;
    for (j = 0; j <= ny-1; j++) {
      s[j] = s[j] + r[i]*A[i+j];
      q[i] = q[i] + A[i+j]*p[j];
    }
  }

  /*@ end @*/
  /*@ end @*/

}
