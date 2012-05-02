void FormJacobian2D(double lambda, int m, int n, double *x, double *dia) {
  int i;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[] = range(32,65,32);
          param CB[] = [True, False];
        }
        def input_params {
          param Lambda = 6;
          param M = 4;
          param N = 4;
        }
        def input_vars {
          decl dynamic double x[M*N] = random;
          decl dynamic double dia[5*M*N-2*M-2] = 0;
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 1;
        }
  ) @*/

  int m = M;
  int n = N;
  double lambda = Lambda;
  double hx     = 1.0/(m-1);
  double hy     = 1.0/(n-1);
  double sc     = hx*hy*lambda;
  double hxdhy  = hx/hy;
  double hydhx  = hy/hx;

  int nrows=m*n;
  int bb = m;
  int be = nrows-m;

  /*@ begin Loop(transform CUDA(threadCount=TC, cacheBlocks=CB)

  for(i=bb; i<=be-1; i++) {
    dia[i-m]           = -hxdhy;
    dia[i+nrows-m-1]   = -hydhx;
    dia[i+2*nrows-m-1] = 2.0*(hydhx + hxdhy) - sc*exp(x[i]);
    dia[i+3*nrows-m-1] = -hydhx;
    dia[i+4*nrows-m-2] = -hxdhy;
  }

  ) @*/

  for(i=bb; i<=be-1; i++) {
    dia[i-m]           = -hxdhy;
    dia[i+nrows-m-1]   = -hydhx;
    dia[i+2*nrows-m-1] = 2.0*(hydhx + hxdhy) - sc*exp(x[i]);
    dia[i+3*nrows-m-1] = -hydhx;
    dia[i+4*nrows-m-2] = -hxdhy;
  }

  /*@ end @*/
  /*@ end @*/
}
