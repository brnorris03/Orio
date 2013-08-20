void VecWAXPY(int n, double *w, double a, double *x, double *y) {

  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param WI[]  = range(32,65,32);
          param WG[]  = [4,8,16,32,64];
          param UI[]  = range(1,3);
          param CB[] = [True, False];
          param CL[] = ['', '-cl-fast-relaxed-math'];
          param SH[]  = [True, False];
          param VH[]  = [0,2,4];
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
          decl static double y[N] = random;
          decl static double w[N] = 0;
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

  /*@ begin Loop(transform OpenCL(workGroups=WG, workItemsPerGroup=WI, unrollInner=UI, cacheBlocks=CB, sizeHint=SH, vecHint=VH, clFlags=CL, device=0)

  for (i=0; i<=n-1; i++)
    w[i]=a*x[i]+y[i];

  ) @*/

  for (i=0; i<=n-1; i++)
    w[i]=a*x[i]+y[i];

  /*@ end @*/
  /*@ end @*/
}
