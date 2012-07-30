void MatMult_SeqSG(double* A, double* x, double* y, int M, int N, int P, int NOS, int DOF) {

  register int i,j;

  int nrows=M*N*P;
  int ndiags=NOS;
  int offsets[NOS]={-M*N*DOF,-M*DOF,-DOF,0,DOF,M*DOF,M*N*DOF};
  int col;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, preferL1Size=16, unrollInner=2)

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
}
