void VecAXPBY(int n, double a, double *x, double b, double *y) {

    register int i;

    /*@ begin PerfTuning (
          def performance_params {
            param TC[] = range(16,513,16);
            param CB[] = [True, False];
            param PHM[] = [False];
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
            decl static double x[N] = random;
            decl static double y[N] = random;
          }
          def performance_counter {
            arg method = 'basic timer';
            arg repetitions = 10;
          }
    ) @*/

    int n=N;

    /*@ begin Loop (
          transform CUDA(threadCount=TC, cacheBlocks=CB, pinHostMem=PHM)
        for (i=0; i<=n-1; i++)
          y[i]=a*x[i]+b*y[i];
    ) @*/

    for (i=0; i<=n-1; i++)
        y[i]=a*x[i]+b*y[i];

    /*@ end @*/
    /*@ end @*/
}
