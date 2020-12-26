void VecWAXPY(int n, double *w, double a, double *x, double *y) {

  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param TC[] = range(32,65,32);
          param BC[] = range(14,29,14);
          param SC[] = range(1,3);
          param CB[] = [True, False];
          param PL[] = [16,48];
          param CFLAGS[] = ['', '-use_fast_math', '-Xptxas -dlcm=cg'];
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20 @CFLAGS';
        }
        def input_params {
          param N[] = [1000];
        }
        def input_vars {
          decl double a = random;
          decl static double x[N] = random;
          decl static double y[N] = random;
          decl static double w[N] = 0;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 1;
        }
  ) @*/

  int n=N;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, streamCount=SC, cacheBlocks=CB, preferL1Size=PL)

  for (i=0; i<=n-1; i++)
    w[i]=a*x[i]+y[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    w[i]=a*x[i]+y[i];

  /*@ end @*/
  /*@ end @*/
}
