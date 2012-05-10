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

/*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, preferL1Size=16, unrollInner=2)

  for(i=0; i<=nrows-1; i++) {
    for(j=0; j<=ndiags-1; j++){
      col = i+offsets[j];
      if(col>=0&&col<nrows)
        y[i] += A[i+j*nrows] * x[col];
    }
  }

  ) @*/
{
  /*declare variables*/
  double *dev_y, *dev_A, *dev_x;
  int *dev_offsets;
  int nthreads=32;
  /*calculate device dimensions*/
  dim3 dimGrid, dimBlock;
  dimBlock.x=nthreads;
  dimGrid.x=14;
  /*allocate device memory*/
  int nbytes=nrows*sizeof(double);
  cudaMalloc((void**)&dev_y,nbytes);
  cudaMalloc((void**)&dev_A,nbytes);
  cudaMalloc((void**)&dev_x,nbytes);
  cudaMalloc((void**)&dev_offsets,sizeof(offsets));
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
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
  orcu_kernel2<<<dimGrid,dimBlock>>>(nrows,ndiags,dev_offsets,dev_y,dev_A,dev_x);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&orcu_elapsed,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  /*copy data from device to host*/
  cudaMemcpy(y,dev_y,nbytes,cudaMemcpyDeviceToHost);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
  /*free allocated memory*/
  cudaFree(dev_y);
  cudaFree(dev_A);
  cudaFree(dev_x);
  cudaFree(dev_offsets);
}
/*@ end @*/
}
__global__ void orcu_kernel2(const int nrows, const int ndiags, int* offsets, double* y, double* A, double* x) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  int j, col;
  for (int i=tid; i<=nrows-1; i+=gsize) {
    {
      #pragma unroll 2
      for (j=0; j<=ndiags-1; j++ ) {
        col=i+offsets[j];
        if (col>=0&&col<nrows) 
          y[i]=y[i]+A[i+j*nrows]*x[col];
      }
    }
  }
}
