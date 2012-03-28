void VecNorm2(int n, double *x, double r) {

    /*@ begin PerfTuning (
          def performance_params {
            param TC[] = range(16,17,16);
            param CB[] = [True, False];
            param PHM[] = [False];
            param SC[] = range(1,3);
          }
          def build {
            arg build_command = 'nvcc -arch=sm_20';
          }
          def input_params {
            param N[] = [1000];
          }
          def input_vars {
            decl double r = 0;
            decl static double x[N] = random;
          }
          def performance_counter {
            arg method = 'basic timer';
            arg repetitions = 5;
          }
    ) @*/

    register int i;
    int n=N;

    /*@ begin Loop (
          transform CUDA(threadCount=TC, cacheBlocks=CB, pinHostMem=PHM, streamCount=SC)
        for (i=0; i<=n-1; i++)
          r=r+x[i]*x[i];
        r=sqrt(r);
    ) @*/

    for (i=0; i<=n-1; i++)
        r=r+x[i]*x[i];
    r=sqrt(r);

    /*@ end @*/
    /*@ end @*/
}
