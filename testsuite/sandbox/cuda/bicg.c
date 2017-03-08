void BICG(double* A, double* p, double* q, double* r, double* s, int NX, int NY) {

  register int i,j;
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
          decl static double p[NY]     = random;
          decl static double q[NX]     = 0;
          decl static double r[NX]     = random;
          decl static double s[NY]     = 0;
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
