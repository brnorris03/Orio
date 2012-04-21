void FormFunction2D(double lambda, int m, int n, double* X, double *F) {
  int i;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,33,32);
          param CB[]  = [False];
          param PHM[] = [False];
        }
        def input_params {
          param m   = 512;
          param n   = 512;
          param Nos = 5;
          param lambda = 6;
        }
        def input_vars {
          decl dynamic double X[m*n*Nos] = random;
          decl dynamic double F[m*n*Nos] = 0;
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 10;
        }
  ) @*/

  int nrows=m*n;
  int offsets[Nos];
  offsets[0]=-m;
  offsets[1]=-1;
  offsets[2]=0;
  offsets[3]=1;
  offsets[4]=m;
  int bb = offsets[4];
  int be = nrows-offsets[4];

  double hx     = 1.0/(m-1);
  double hy     = 1.0/(n-1);
  double sc     = hx*hy*lambda;
  double hxdhy  = hx/hy;
  double hydhx  = hy/hx;

  /*@ begin Loop(transform CUDA(threadCount=TC, cacheBlocks=CB, pinHostMem=PHM)
  for(i=bb; i<=be-1; i++) {
    F[i] = (2*X[i+2*nrows] - X[i+nrows] - X[i+3*nrows])*hydhx + (2*X[i+2*nrows] - X[i] - X[i+4*nrows])*hxdhy - sc*exp(X[i+2*nrows]);
  }
  ) @*/
  for(i=bb; i<=be-1; i++) {
    F[i] = (2*X[i+2*nrows] - X[i+nrows] - X[i+3*nrows])*hydhx + (2*X[i+2*nrows] - X[i] - X[i+4*nrows])*hxdhy - sc*exp(X[i+2*nrows]);
  }
  /*@ end @*/
  /*@ end @*/
}
