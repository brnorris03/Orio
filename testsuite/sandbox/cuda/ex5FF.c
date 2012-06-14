void FormFunction2D(double lambda, int m, int n, double* X, double *F) {

  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param TC[] = range(32,1025,32);
          param BC[] = range(14,113,14);
          param SC[] = range(1,6);
          param PL[] = [16,48];
          param CFLAGS[] = map(join, product(['', '-O1', '-O2', '-O3']));
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20 @CFLAGS';
        }
        def input_params {
          param lambda = 6;
          param M[] = [4,8,16,32];
          param N[] = [4,8,16,32];
          constraint c1 = (M==N);
        }
        def input_vars {
          decl dynamic double X[M*N] = random;
          decl dynamic double F[M*N] = 0;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 6;
        }
  ) @*/

  int m        = M;
  int n        = N;
  int nrows    = m*n;
  double hx    = 1.0/(m-1);
  double hy    = 1.0/(n-1);
  double sc    = hx*hy*lambda;
  double hxdhy = hx/hy;
  double hydhx = hy/hx;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, streamCount=SC, preferL1Size=PL)

  for(i=0; i<=nrows-1; i++) {
    if (i<m || i>=nrows-m || i%m==0 || i%m==m-1) {
      F[i] = X[i];
    } else {
      F[i] = (2*X[i] - X[i-1] - X[i+1])*hydhx
           + (2*X[i] - X[i-m] - X[i+m])*hxdhy
           - sc*exp(X[i]);
    }
  }

  ) @*/

  for(i=0; i<=nrows-1; i++) {
    if (i<m || i>=nrows-m || i%m==0 || i%m==m-1) {
      F[i] = X[i];
    } else {
      F[i] = (2*X[i] - X[i-1] - X[i+1])*hydhx
           + (2*X[i] - X[i-m] - X[i+m])*hxdhy
           - sc*exp(X[i]);
    }
  }

  /*@ end @*/
  /*@ end @*/
}
