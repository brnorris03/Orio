void MatMult_SeqDIA(double* A, double* x, double* y, int M, int N, int P, int NOS, int DOF) {

  register int i,j,k;
  int col,row;
  double ysum;
  /*@ begin PerfTuning (
        def performance_params {
          param TC[]  = range(32,1025,32);
          param BC[]  = range(13,105,13);
          param PL[]  = [16,32,48];
        }
        def input_params {
          param M[] = [32,64,96];
          param N[] = [32,64,96];
          param P[] = [32,64,96];
          param NOS = 7;
          param DOF[] = range(1,17);
          constraint c1 = (M==N);
          constraint c2 = (N==P);
        }
        def input_vars {
          decl dynamic double A[M*N*P*DOF*DOF*NOS] = random;
          decl dynamic double x[M*N*P*DOF]         = random;
          decl dynamic double y[M*N*P*DOF]         = 0;
          decl static  int offsets[NOS]            = {-M*N*DOF,-M*DOF,-DOF,0,DOF,M*DOF,M*N*DOF};
        }
  ) @*/

  int nrows=M*N*P*DOF;
  int ndiags=NOS;
  int ndofs=DOF;
  int sbdiag=M*N*P*DOF*DOF;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL)

  for(i=0; i<=nrows-1; i++){
    ysum = 0.0;
    for(j=0; j<=ndiags-1; j++){
      row = i+j*sbdiag;
      col = (floor((float)i/ndofs)+offsets[j])*ndofs;
      if(col>=0&&col<nrows)
        for(k=0; k<=ndofs-1; k++)
          ysum += A[row+k*nrows] * x[col+k];
    }
    y[i] = ysum;
  }

  ) @*/

  for(i=0; i<=nrows-1; i++){
    ysum = 0.0;
    for(j=0; j<=ndiags-1; j++){
      row = i+j*sbdiag;
      col = (floor((float)i/ndofs)+offsets[j])*ndofs;
      if(col>=0&&col<nrows)
        for(k=0; k<=ndofs-1; k++)
          ysum += A[row+k*nrows] * x[col+k];
    }
    y[i] = ysum;
  }

  /*@ end @*/
  /*@ end @*/
}

