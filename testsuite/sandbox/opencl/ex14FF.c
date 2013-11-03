void FormFunction3D(double lambda, int m, int n, int p, double* X, double *F) {
  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param WI[] = [32,64,128,256];
          param WG[] = [16,32,64,128];
          param CFLAGS[] = ['','-cl-fast-relaxed-math'];
          param UIF[] = [1,2,4];
          param SH[]  = [False,True];
          param VH[]  = [0,2,4];
        }
        def build {
          arg build_command = 'gcc -O3 -I/opt/AMDAPP/include -I/home/users/nchaimov/werewolf/tau2/include -L/home/users/nchaimov/werewolf/tau2/x86_64/lib -lTAU -lOpenCL -DPROFILING_ON -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -fPIC -DTAU_SS_ALLOC_SUPPORT -DTAU_STRSIGNAL_OK';
        }
        def input_params {
          param lambda = 6;
          param M[] = [4,8,16,32];
          param N[] = [4,8,16,32];
          param P[] = [4,8,16,32];
          constraint c1 = (M==N) and (N==P);
        }
        def input_vars {
          decl dynamic double X[M*N*P] = random;
          decl dynamic double F[M*N*P] = 0;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 6;
        }
        def performance_test_code {
          arg skeleton_code_file = 'tau_skeleton.c';
        }
        def search {
          arg algorithm = 'Simplex';
          arg time_limit = 30;
          arg total_runs = 10;
        }
  ) @*/

  int m          = M;
  int n          = N;
  int p          = P;
  int nrows      = m*n*p;
  double hx      = 1.0/(m-1);
  double hy      = 1.0/(n-1);
  double hz      = 1.0/(p-1);
  double sc      = hx*hy*hz*lambda;
  double hxhzdhy = hx*hz/hy;
  double hyhzdhx = hy*hz/hx;
  double hxhydhz = hx*hy/hz;

  /*@ begin Loop(transform OpenCL(workGroups=WG, workItemsPerGroup=WI, clFlags=CFLAGS, unrollInner=UIF, sizeHint=SH, vecHint=VH, device=0)

  for(i=0; i<=nrows-1; i++) {
    if (i<m*n || i>=nrows-m*n || i%(m*n)<m || i%(m*n)>=m*n-m || i%m==0 || i%m==m-1) {
      F[i] = X[i];
    } else {
      F[i] = (2*X[i] - X[i-1  ] - X[i+1  ])*hyhzdhx
           + (2*X[i] - X[i-m  ] - X[i+m  ])*hxhzdhy
           + (2*X[i] - X[i-m*n] - X[i+m*n])*hxhydhz
           - sc*exp(X[i]);
    }
  }

  ) @*/
  for(i=0; i<=nrows-1; i++) {
    if (i<m*n || i>=nrows-m*n || i%(m*n)<m || i%(m*n)>=m*n-m || i%m==0 || i%m==m-1) {
      F[i] = X[i];
    } else {
      F[i] = (2*X[i] - X[i-1  ] - X[i+1  ])*hyhzdhx
           + (2*X[i] - X[i-m  ] - X[i+m  ])*hxhzdhy
           + (2*X[i] - X[i-m*n] - X[i+m*n])*hxhydhz
           - sc*exp(X[i]);
    }
  }
  /*@ end @*/
  /*@ end @*/
}
