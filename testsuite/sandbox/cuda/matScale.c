void MatScale_SeqDIA(double* A, double a) {

  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(16,33,16);
          param CB[]  = [True, False];
          param SC[]  = range(1,3);
        }
        def input_params {
          param N[] = [1000];
        }
        def input_vars {
          decl dynamic double A[N] = random;
          decl double a = random;
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 10;
        }
  ) @*/

  register int i;
  int nz = N;

  /*@ begin Loop(transform CUDA(threadCount=TC, cacheBlocks=CB, streamCount=SC)

  for(i=0;i<=nz-1;i++) {
    A[i] *= a;
  }

  ) @*/

  for(i=0;i<=nz-1;i++) {
    A[i] *= a;
  }

  /*@ end @*/
  /*@ end @*/
}
