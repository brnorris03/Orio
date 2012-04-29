void VecAXPBYPCZ(int n, double a, double *x, double b, double *y, double c, double *z) {

    /*@ begin PerfTuning (
          def performance_params {
            param TC[] = range(16,33,16);
            param CB[] = [True, False];
            param SC[] = range(1,3);
          }
          def build {
            arg build_command = 'nvcc -arch=sm_20';
          }
          def input_params {
            param N[] = [1000];
          }
          def input_vars {
            decl double a = random;
            decl double b = random;
            decl double c = random;
            decl static double x[N] = random;
            decl static double y[N] = random;
            decl static double z[N] = random;
          }
          def performance_counter {
            arg method = 'basic timer';
            arg repetitions = 10;
          }
    ) @*/

    register int i;
    int n=N;

    /*@ begin Loop(transform CUDA(threadCount=TC, cacheBlocks=CB, streamCount=SC)
        for (i=0; i<=n-1; i++)
          y[i]=a*x[i]+b*y[i]+c*z[i];
    ) @*/

    for (i=0; i<=n-1; i++)
      y[i]=a*x[i]+b*y[i]+c*z[i];

    /*@ end @*/
    /*@ end @*/
}
