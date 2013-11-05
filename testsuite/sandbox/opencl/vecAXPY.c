void VecAXPY(int n, double a, double *x, double *y) {

  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param WI[] = [32,64,128,256];
          param WG[] = [4,8,16,32,64];
          param CB[] = [True, False];
          param SH[] = [True, False];
          param UI[] = range(1,4);
          param VH[] = [0,2,4];
          param CL[] = ['-cl-fast-relaxed-math'];
        }
        def build {
          arg build_command = 'gcc -O3 -I/opt/AMDAPP/include -I/home/users/nchaimov/werewolf/tau2/include -L/home/users/nchaimov/werewolf/tau2/x86_64/lib -lTAU -lOpenCL -DPROFILING_ON -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -fPIC -DTAU_SS_ALLOC_SUPPORT -DTAU_STRSIGNAL_OK';
        }
        def input_params {
          param N[] = [1000000];
        }
        def input_vars {
          decl double a = random;
          decl static double x[N] = random;
          decl static double y[N] = 0;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 3;
        }
        def performance_test_code {
          arg skeleton_code_file = 'tau_skeleton.c';
        }
  ) @*/

  int n=N;

  /*@ begin Loop(transform OpenCL(workGroups=WG, workItemsPerGroup=WI, sizeHint=SH, vecHint=VH, cacheBlocks=CB, unrollInner=UI, clFlags=CL, device=1)

  for (i=0; i<=n-1; i++)
    y[i]+=a*x[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    y[i]+=a*x[i];

  /*@ end @*/
  /*@ end @*/
}
