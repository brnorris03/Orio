void axpy5(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4, double a5, double *x5) {
    
    register int i;
    
    /*@ begin PerfTuning (
          def performance_params {
            param TC[] = range(1,1024);
          }
          def build {
            arg build_command = 'nvcc -arch=sm_20';
          }
          def input_params {
            param N[] = [1000];
          }
          def input_vars {
            decl static double y[N] = 0;
            decl double a1 = random;
            decl double a2 = random;
            decl double a3 = random;
            decl double a4 = random;
            decl double a5 = random;
            decl static double x1[N] = random;
            decl static double x2[N] = random;
            decl static double x3[N] = random;
            decl static double x4[N] = random;
            decl static double x5[N] = random;
          }
          def performance_counter {
            arg method = 'basic timer';
            arg repetitions = 10;
          }
    ) @*/

    int n=N;

    /*@ begin Loop (
          transform CUDA(
            threadCount=TC,
            maxBlocks=65535
          )
        for (i=0; i<=n-1; i++)
          y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
    ) @*/

    for (i=0; i<=n-1; i++)
        y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];

    /*@ end @*/
    /*@ end @*/
}
