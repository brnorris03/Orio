void MatMult_SeqSG(double* A, double* x, double* y, int m, int n, int p, int nos, int dof) {

  register int i,j;

  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,33,32);
          param CB[]  = [True];
          param PHM[] = [False];
          param SC[]  = range(1,2);
        }
        def input_params {
          param m[]   = [2];
          param n[]   = [2];
          param p[]   = [2];
          param Nos[] = [7];
          param dof[] = [1];
        }
        def input_vars {
          decl static double A[m*n*p*Nos*dof] = random;
          decl static double x[m*n*p*dof]     = random;
          decl static double y[m*n*p*dof]     = 0;
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20';
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 10;
        }
  ) @*/

  int nrows=m*n*p;
  int ndiags=Nos;
  int offsets[ndiags];
  offsets[0]=-m*n*dof;
  offsets[1]=-m*dof;
  offsets[2]=-dof;
  offsets[3]=0;
  offsets[4]=dof;
  offsets[5]=m*dof;
  offsets[6]=m*n*dof;

  /*@
    begin Loop(
      transform CUDA(threadCount=TC, cacheBlocks=CB, pinHostMem=PHM, streamCount=SC)
        for(i=0; i<=nrows-1; i++)
          for(j=0; j<=ndiags-1; j++)
            y[i] += A[i+j*nrows] * x[i+offsets[j]];
    )
  @*/

  /*@ end @*/
  /*@ end @*/
}
