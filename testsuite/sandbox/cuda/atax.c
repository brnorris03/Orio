void ATAX(double* A, double* x, double* y, int NX, int NY) {

  register int i,j,k;
  int tmp;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,1025,32); 	# threads per block
          param BC[]  = range(108,1081,108);	# number of blocks, grid.x
          param ILUF[] = range(1,13);		# inner loop unroll factor
          param PL[]  = [16,32,48]; 		# preferred L1 cache size
          param SC[]  = [1];			# number of streams
          param CFLAGS[] =['-O0','-O3','-use_fast_math']; # -O3 is default for nvcc
        }
        def input_params {
          param NX[] = [1024];
          param NY[] = [1024];
          constraint c1 = (NX==NY);
        }
        def input_vars {
          decl static double A[NX*NY] = random;
          decl static double x[NY]     = random;
          decl static double y[NY]     = 0;
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

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL, 
  				unrollInner=ILUF, streamCount=SC)
  for (i = 0; i<=nx-1; i++) {
    tmp = 0;
    for (k = 0; k<=ny-1; k++) 
      tmp = tmp + A[i+k]*x[k];
    for (j = 0; j<=ny-1; j++) 
      y[j] = y[j] + A[i+j]*tmp;
   }
   ) @*/

  for (i = 0; i<=nx-1; i++) {
    tmp = 0;
    for (k = 0; k<=ny-1; k++) 
      tmp = tmp + A[i+k]*x[k];
    for (j = 0; j<=ny-1; j++) 
      y[j] = y[j] + A[i+j]*tmp;
  }

  
  /*@ end @*/
  /*@ end @*/

}
