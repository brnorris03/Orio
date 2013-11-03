void VecSum(int n, double *x, double s) {

  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param WI[] = [1,2,4,6,8,10,12];
          param WG[] = [32,64];
          param SH[] = [False,True];
        }
        def build {
          arg build_command = 'gcc -g3 -framework OpenCL -I/Users/nchaimov/src/tau2/include -L/Users/nchaimov/src/tau2/apple/lib -lTAU -DPROFILING_ON -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -fPIC -DTAU_SS_ALLOC_SUPPORT -DTAU_STRSIGNAL_OK';
        }
        def input_params {
          param N[] = [1000];
        }
        def input_vars {
          decl double s = 0;
          decl static double x[N] = random;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 10;
        }
        def performance_test_code {
          arg skeleton_code_file = 'tau_skeleton.c';
        }
  ) @*/
  int n=N;

  /*@ begin Loop(transform OpenCL(workGroups=WG, workItemsPerGroup=WI, sizeHint=SH)

  for (i=0; i<=n-1; i++)
    s+=x[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    s+=x[i];

  /*@ end @*/
  /*@ end @*/
}
