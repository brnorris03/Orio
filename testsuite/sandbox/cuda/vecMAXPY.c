void VecMAXPY(int n, int nv, double *a, double *x, double *y) {

  /*@ begin PerfTuning (
        def performance_params {
          param TC[] = range(16,33,16);
          param CB[] = [True, False];
          param UIF[] = range(1,3);
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20';
        }
        def input_params {
          param N[] = [10];
          param NV[] = [16];
        }
        def input_vars {
          decl static double y[N] = 0;
          decl static double a[NV] = random;
          decl static double x[NV*N] = random;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 10;
        }
  ) @*/

  register int i,j;
  int n=N;
  int nv=NV;

  /*@ begin Loop (transform CUDA(threadCount=TC, cacheBlocks=CB, unrollInner=UIF)

  for (i=0; i<=n-1; i++)
    for (j=0; j<=nv-1; j++)
      y[i]+=a[j]*x[j*n+i];

  ) @*/

  for (i=0; i<=n-1; i++)
    for (j=0; j<=nv-1; j++)
      y[i]+=a[j]*x[j*n+i];

  /*@ end @*/
  /*@ end @*/
}
