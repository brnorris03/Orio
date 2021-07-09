void FormJacobian3D(double lambda, int m, int n, int p, double *x, double *dia) {
  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param TC[] = range(32,1025,32);
          param BC[] = range(14,113,14);
          param PL[] = [16,48];
          param CFLAGS[] = ['', '-O1', '-O2', '-O3'];
        }
        def build {
          arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
        }
        def input_params {
          param Nos = 7;
          param lambda = 6;
          param M[] = [4,8,16,32];
          param N[] = [4,8,16,32];
          param P[] = [4,8,16,32];
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

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL)

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
