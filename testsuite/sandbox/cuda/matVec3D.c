void MatMult_SeqDIA(double* A, double* x, double* y, int M, int N, int P, int NOS, int DOF) {

  register int i,j;
  int col;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,1025,32);
          param BC[]  = range(14,113,14);
          param UIF[] = range(1,8);
          param PL[]  = [16,48];
          param CFLAGS[] = ['', '-O1', '-O2', '-O3'];
        }
        def input_params {
          param M[] = [16,32,64,128,256];
          param N[] = [16,32,64,128,256];
          param P[] = [16,32,64,128,256];
          param NOS = 7;
          param DOF = 1;
          constraint c1 = (M==N);
          constraint c2 = (N==P);
        }
        def input_vars {
          decl static double A[M*N*P*NOS*DOF] = random;
          decl static double x[M*N*P*DOF]     = random;
          decl static double y[M*N*P*DOF]     = 0;
          decl static int offsets[NOS]        = {-M*N*DOF,-M*DOF,-DOF,0,DOF,M*DOF,M*N*DOF};
        }
        def build {
          arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 6;
        }
  ) @*/

  int nrows=M*N*P;
  int ndiags=NOS*DOF;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL, unrollInner=UIF)

  for(i=0; i<=nrows-1; i++) {
    for(j=0; j<=ndiags-1; j++){
      col = i+offsets[j];
      if(col>=0&&col<nrows)
        y[i] += A[i+j*nrows] * x[col];
    }
  }

  ) @*/

  for(i=0; i<=nrows-1; i++) {
    for(j=0; j<=ndiags-1; j++){
      col = i+offsets[j];
      if(col>=0&&col<nrows)
        y[i] += A[i+j*nrows] * x[col];
    }
  }

  /*@ end @*/
  /*@ end @*/
}
