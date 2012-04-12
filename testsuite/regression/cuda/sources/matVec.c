void MatMult_SeqSG(double* A, double* x, double* y, int m, int n, int p, int nos, int dof) {


  register int i,j;
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
  int col;

  /*@
      begin Loop(
      transform CUDA(threadCount=16, cacheBlocks=False, pinHostMem=False)
        for(i=0; i<=nrows-1; i++) {
          for(j=0; j<=ndiags-1; j++){
            col = i+offsets[j];
            if(col>=0&&col<nrows)
              y[i] += A[i+j*nrows] * x[col];
          }
        }
    )
  @*/

  /*@ end @*/
}
