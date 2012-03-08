void MatVec_StencilSG(int n, double* A, double* x, double* y) {

  register int s;

  /*@
    begin Loop(
      transform CUDA(threadCount=16, cacheBlocks=False, pinHostMem=False, streamCount=1, domain='Stencil_SG3_Star1_Dof1')
        for(s=0; s<=n-1; s++)
          y[s] += A[s] * x[s];
    )
  @*/

  /*@ end @*/
}
