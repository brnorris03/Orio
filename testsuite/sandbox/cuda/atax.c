void ATAX(double* A, double* x, double* y, int NX, int NY) {

  register int i,j,k;
  int tmp;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,1025,32);
          param BC[]  = range(16,129,16);
          param UIF[] = range(1,6);
          param PL[]  = [16,48];
	    param SC[]  = range(1,6);
          param CFLAGS[] = map(join, product(['', '-use_fast_math']));
        }
        def input_params {
          param NX[] = [32];
          param NY[] = [32];
          constraint c1 = (NX==NY);
        }
        def input_vars {
          decl static double A[NX*NY] = random;
          decl static double x[NY]     = random;
          decl static double y[NY]     = 0;
        }
        def build {
          arg build_command = 'nvcc -g -lineinfo -arch=sm_20 -O3 @CFLAGS';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 10;
        }
	) @*/

  int nx=NX;
  int ny=NY;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL, unrollInner=UIF, streamCount=SC)
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
