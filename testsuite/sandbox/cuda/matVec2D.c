void MatMult_SeqSG(double* A, double* x, double* y, int m, int n, int p, int nos, int dof) {

  register int i,j;
  int col;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,1025,32);
          param BC[]  = range(14,113,14);
          param UIF[] = range(1,6);
          param PL[]  = [16,48];
          param CFLAGS[] = map(join, product(['', '-use_fast_math'], ['', '-Xptxas -dlcm=cg']));
        }
        def input_params {
          param m[]   = [32,64,128,256,512];
          param n[]   = [32,64,128,256,512];
          param Nos[] = [5];
          param dof[] = [1];
          constraint c1 = (m==n);
        }
        def input_vars {
          decl static double A[m*n*Nos*dof] = random;
          decl static double x[m*n*dof]     = random;
          decl static double y[m*n*dof]     = 0;
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20 @CFLAGS';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 6;
        }
  ) @*/

  int nrows=m*n;
  int ndiags=Nos;
  int offsets[ndiags];
  offsets[0]=-m*dof;
  offsets[1]=-dof;
  offsets[2]=0;
  offsets[3]=dof;
  offsets[4]=m*dof;

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
