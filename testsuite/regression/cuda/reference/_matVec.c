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
  {
    /*declare variables*/
    double *dev_y, *dev_A, *dev_x;
    int *dev_offsets;
    int nthreads=16;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=(nrows+nthreads-1)/nthreads;
    /*allocate device memory*/
    int nbytes=nrows*sizeof(double);
    cudaMalloc((void**)&dev_y,nbytes);
    cudaMalloc((void**)&dev_A,nbytes);
    cudaMalloc((void**)&dev_x,nbytes);
    cudaMalloc((void**)&dev_offsets,sizeof(offsets));
    /*copy data from host to device*/
    cudaMemcpy(dev_y,y,nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A,A,nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x,x,nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_offsets,offsets,sizeof(offsets),cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    orcu_kernel3<<<dimGrid,dimBlock>>>(nrows,ndiags,dev_offsets,dev_y,dev_A,dev_x);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /*copy data from device to host*/
    cudaMemcpy(y,dev_y,nbytes,cudaMemcpyDeviceToHost);
    /*free allocated memory*/
    cudaFree(dev_y);
    cudaFree(dev_A);
    cudaFree(dev_x);
    cudaFree(dev_offsets);
  }
/*@ end @*/
}
__global__ void orcu_kernel3(int nrows, int ndiags, int* offsets, double* y, double* A, double* x) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  int j, col;
  if (tid<=nrows-1) {
    {
      #pragma unroll
      for (j=0; j<=ndiags-1; j++ ) {
        col=tid+offsets[j];
        if (col>=0&&col<nrows) 
          y[tid]=y[tid]+A[tid+j*nrows]*x[col];
      }
    }
  }
}
