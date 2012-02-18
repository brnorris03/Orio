void VecWAXPY(int n, double *w, double a, double *x, double *y) {

    register int i;

    /*@ begin PerfTuning (
          def performance_params {
            param TC[] = range(16,513,16);
            param CB[] = [True, False];
          }
          def build {
            arg build_command = 'nvcc -arch=sm_20';
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
            arg repetitions = 10;
          }
    ) @*/

    int n=N;

    /*@ begin Loop (
          transform CUDA(threadCount=TC, cacheBlocks=CB)
        for (i=0; i<=n-1; i++)
          w[i]=a*x[i]+y[i];
    ) @*/

    for (i=0; i<=n-1; i++)
        w[i]=a*x[i]+y[i];

    /*@ end @*/
    /*@ end @*/
}
