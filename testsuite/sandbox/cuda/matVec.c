void MatVec_StencilSG(int n, double* A, double* x, double* y) {

  /*@ begin PerfTuning (
        def performance_params {
          param TC[] = range(16,17,16);
          param CB[] = [False];
          param PHM[] = [False];
          param SC[] = range(1,2);
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20';
        }
        def input_params {
          param N[] = [125];
        }
        def input_vars {
          decl static double A[N] = random;
          decl static double x[N] = random;
          decl static double y[N] = 0;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 10;
        }
  ) @*/

  register int s;
  int n=N;

  /*@
    begin Loop(
      transform CUDA(threadCount=TC, cacheBlocks=CB, pinHostMem=PHM, streamCount=SC, domain='Stencil_SG3_Star1_Dof1')
        for(s=0; s<=n-1; s++)
          y[s] += A[s] * x[s];
    )
  @*/

  /*@ end @*/
  /*@ end @*/
}
