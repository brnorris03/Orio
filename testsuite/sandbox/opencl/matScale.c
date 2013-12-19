void MatScale_SeqDIA(double* A, double a) {

  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param WI[] = [16, 32, 64, 128];
          param WG[] = [4, 8, 16, 32];
          param SH[] = [False, True];
          param VH[] = [0,1,2,4];
          param CB[] = [False, True];
          param UI[] = range(1,6);
          param CLFLAGS[] = ['','-cl-fast-relaxed-math'];
        }
        def build {
          arg build_command = 'gcc -g3 -framework OpenCL -I/Users/nchaimov/src/tau2/include -L/Users/nchaimov/src/tau2/apple/lib -lTAU -DPROFILING_ON -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -fPIC -DTAU_SS_ALLOC_SUPPORT -DTAU_STRSIGNAL_OK';
        }
        def input_params {
          param N[] = [100000];
        }
        def input_vars {
          decl dynamic double A[N] = random;
          decl double a = random;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 10;
        }
        def performance_test_code {
          arg skeleton_code_file = 'tau_skeleton.c';
        }
  ) @*/

  int nz = N;

  /*@ begin Loop(transform OpenCL(workGroups=WG, workItemsPerGroup=WI, sizeHint=SH, vecHint=VH, cacheBlocks=CB, unrollInner=UI, clFlags=CLFLAGS)

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
