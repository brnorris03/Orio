void VecMAXPY(int n, int nv, double *a, double *x, double *y) {

  register int i,j;
  /*@ begin PerfTuning(
        def performance_params {
          param TC[]  = range(32,65,32);
          param BC[]  = range(14,29,14);
          param UIF[] = range(1,3);
          param PL[]  = [16,48];
          param CFLAGS[] = map(join, product(['', '-use_fast_math'], ['', '-Xptxas -dlcm=cg']));
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20 @CFLAGS';
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
          arg repetitions = 1;
        }
  ) @*/

  int n=N;
  int nv=NV;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL, unrollInner=UIF)

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
