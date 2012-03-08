void MatVec_StencilSG(int n, double* A, double* x, double* y) {

  register int s;

  /*@
    begin Loop(
      transform CUDA(threadCount=16, cacheBlocks=False, pinHostMem=False, streamCount=1, domain='Stencil_SG3_Star1_Dof1')
        for(s=0; s<=n-1; s++)
          y[s] += A[s] * x[s];
    )
  @*/
  {
    /*declare variables*/
    double *dev_y, *dev_A, *dev_x;
    int nthreads=16;
    dim3 dimGrid, dimBlock;
    int orcu_i;
    /*calculate device dimensions*/
    dimGrid.x=ceil((float)n/(float)nthreads);
    dimBlock.x=nthreads;
    /*allocate device memory*/
    int scSize=n*sizeof(double);
    cudaMalloc((void**)&dev_y,scSize);
    cudaMalloc((void**)&dev_A,scSize);
    cudaMalloc((void**)&dev_x,scSize);
    /*copy data from host to device*/
    cudaMemcpy(dev_y,y,scSize,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A,A,scSize,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x,x,scSize,cudaMemcpyHostToDevice);
    /*stencil domain parameters*/
    int gm=round(pow(n,(double)1/3));
    int gn=gm;
    int dof=1;
    int nos=7;
    int sidx[nos];
    sidx[0]=0;
    sidx[1]=dof;
    sidx[2]=gm*dof;
    sidx[3]=gm*gn*dof;
    sidx[4]=-dof;
    sidx[5]=-gm*dof;
    sidx[6]=-gm*gn*dof;
    cudaMemset(y,0,scSize);
    dimGrid.x=n;
    dimBlock.x=nos;
    /*invoke device kernel*/
    orcu_kernel3<<<dimGrid,dimBlock>>>(n,dev_y,dev_A,dev_x,sidx);
    /*copy data from device to host*/
    cudaMemcpy(y,dev_y,scSize,cudaMemcpyDeviceToHost);
    /*free allocated memory*/
    cudaFree(dev_y);
    cudaFree(dev_A);
    cudaFree(dev_x);
  }
/*@ end @*/
}
__global__ void orcu_kernel3(int n, double* y, double* A, double* x, int* sidx) {
  int tid=blockIdx.x+sidx[threadIdx.x];
  if (tid>=0&&tid<n) {
    y[tid]=y[tid]+A[tid]*x[tid];
  }
}
