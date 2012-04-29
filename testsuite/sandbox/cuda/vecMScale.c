void VecScaleMult(int n, double a, double *x) {

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
            decl static double x[N] = random;
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
          x[i]*=a;
    ) @*/

    for (i=0; i<=n-1; i++)
        x[i]*=a;

    /*@ end @*/
    /*@ end @*/
}
