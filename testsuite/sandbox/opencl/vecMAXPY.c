void VecMAXPY(int n, int nv, double *a, double *x, double *y) {

  register int i,j;
  /*@ begin PerfTuning(
        def performance_params {
          param WI[]  = range(32,65,32);
          param WG[]  = [4,8,16,32,64];
          param UI[]  = range(1,3);
          param SH[]  = [True, False];
          param VH[]  = [0,2,4];
          param CL[]  = ['','-cl-fast-relaxed-math'];
        }
        def build {
          arg build_command = 'gcc -O3 -I/opt/AMDAPP/include -I/home/users/nchaimov/werewolf/tau2/include -L/home/users/nchaimov/werewolf/tau2/x86_64/lib -lTAU -lOpenCL -DPROFILING_ON -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -fPIC -DTAU_SS_ALLOC_SUPPORT -DTAU_STRSIGNAL_OK';
        }
        def input_params {
          param N[] = [100000];
          param NV[] = [16];
        }
        def input_vars {
          decl static double y[N] = 0;
          decl static double a[NV] = random;
          decl static double x[NV*N] = random;
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
  int nv=NV;

  /*@ begin Loop(transform OpenCL(workGroups=WG, workItemsPerGroup=WI, sizeHint=SH, vecHint=VH, unrollInner=UI, clFlags=CL, device=1)

  for (i=0; i<=n-1; i++)
    for (j=0; j<=nv-1; j++)
      y[i]+=a[j]*x[j*n+i];

  ) @*/

  for (i=0; i<=n-1; i++)
    for (j=0; j<=nv-1; j++)
      y[i]+=a[j]*x[j*n+i];

  /*@ end @*/
  /*@ end @*/
}
