void FormJacobian3D(double lambda, int m, int n, int p, double *x, double *dia) {
  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param TC[] = range(32,1025,32);
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
          param Nos = 7;
          param lambda = 6;
          param M[] = [128,100,75,64];
          param N[] = [128,100,75,64];
          param P[] = [128,100,75,64];
          constraint c1 = (M==N) and (N==P);
        }
        def input_vars {
          decl dynamic double x[M*N*P] = random;
          decl dynamic double dia[Nos*M*N*P] = 0;
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

  /*@ begin Loop(transform OpenCL(workGroups=WG, workItemsPerGroup=WI, clFlags=CFLAGS, unrollInner=UIF, sizeHint=SH, vecHint=VH, device=1)

  for(i=0; i<=nrows-1; i++) {
    if (i<m*n || i>=nrows-m*n || i%(m*n)<m || i%(m*n)>=m*n-m || i%m==0 || i%m==m-1) {
      dia[i] = 1.0;
    } else {
      dia[i        ] = -hxhydhz;
      dia[i+  nrows] = -hxhzdhy;
      dia[i+2*nrows] = -hyhzdhx;
      dia[i+3*nrows] = 2.0*(hyhzdhx+hxhzdhy+hxhydhz) - sc*exp(x[i]);
      dia[i+4*nrows] = -hyhzdhx;
      dia[i+5*nrows] = -hxhzdhy;
      dia[i+6*nrows] = -hxhydhz;
    }
  }

  ) @*/
  for(i=0; i<=nrows-1; i++) {
    if (i<m*n || i>=nrows-m*n || i%(m*n)<m || i%(m*n)>=m*n-m || i%m==0 || i%m==m-1) {
      dia[i] = 1.0;
    } else {
      dia[i        ] = -hxhydhz;
      dia[i+  nrows] = -hxhzdhy;
      dia[i+2*nrows] = -hyhzdhx;
      dia[i+3*nrows] = 2.0*(hyhzdhx+hxhzdhy+hxhydhz) - sc*exp(x[i]);
      dia[i+4*nrows] = -hyhzdhx;
      dia[i+5*nrows] = -hxhzdhy;
      dia[i+6*nrows] = -hxhydhz;
    }
  }
  /*@ end @*/
  /*@ end @*/
}
