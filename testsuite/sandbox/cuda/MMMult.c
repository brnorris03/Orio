void MatMatMult(double* A, double* B, double* C, int m, int n, int p) {
  register int i,j,k;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,65,32);
          param BC[]  = range(14,28,14);
          param UIF[] = range(1,3);
          param PL[]  = [16,48];
          param CFLAGS[] = ['-O3'];
        }
        def input_params {
          param M[] = [4,8,16,32];
          param N[] = [4,8,16,32];
          param P[] = [4,8,16,32];
        }
        def input_vars {
          decl static double A[M*N] = random;
          decl static double B[N*P] = random;
          decl static double C[M*P] = 0;
        }
        def build {
          arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 1;
        }
  ) @*/

  double s;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL, unrollInner=UIF)

  for(i=0; i<=m-1; i++){
    for(j=0; j<=p-1; j++){
      s = 0.0;
      for(k=0; k<=n-1; k++){
        s += A[i*n+k]*B[k*p+j];
      }
      C[i*n+j]=s;
    }
  }
  ) @*/

  for(i=0; i<=m-1; i++){
    for(j=0; j<=p-1; j++){
      s = 0.0;
      for(k=0; k<=n-1; k++){
        s += A[i*n+k]*B[k*p+j];
      }
      C[i*n+j]=s;
    }
  }
  /*@ end @*/
  /*@ end @*/
}
