void VecSum(int n, double *x, double s) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=1024, maxBlocks=65535)
        for (i=0; i<=n-1; i++)
          s=s+x[i];
    ) @*/
    {
      /*declare device variables*/
      double *dev_x, *dev_s, *dev_block_r, *block_r;
      int* dev_n;
      dim3 dimGrid, dimBlock;
      /*calculate device dimensions*/
      dimGrid.x = ceil((float)n/(float)1024);
      dimBlock.x = 1024;
      /*allocate device memory*/
      cudaMalloc((void**)&dev_n,sizeof(int));
      cudaMalloc((void**)&dev_s,sizeof(double));
      cudaMalloc((void**)&dev_x,n*sizeof(double));
      cudaMalloc((void**)&dev_block_r,dimGrid.x*sizeof(double));
      block_r = (double*)malloc(dimGrid.x*sizeof(double));
      /*copy data from host to devices*/
      cudaMemcpy(dev_n,&n,sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_s,&s,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_x,x,n*sizeof(double),cudaMemcpyHostToDevice);
      /*invoke device kernel function*/
      orcuda_kern_1<<<dimGrid,dimBlock>>>(dev_n,dev_x,dev_s,dev_block_r);
      /*copy data from devices to host*/
      cudaMemcpy(s,dev_s,sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(block_r,dev_block_r,dimGrid.x*sizeof(double),cudaMemcpyDeviceToHost);
      /*post-processing on the host*/
      int i;
      for (i=0; i<dimGrid.x; i++ ) 
        s = s+block_r[i];
      /*free device memory*/
      cudaFree(dev_n);
      cudaFree(dev_x);
      cudaFree(dev_s);
      cudaFree(dev_block_r);
      free(block_r);
    }
/*@ end @*/
}
__global__ void orcuda_kern_1(int* n, double* x, double* s, double* block_r) {
  int tid;
  tid = blockIdx.x*blockDim.x+threadIdx.x;
  if (tid<=(*n)-1) 
    (*s)=(*s)+x[tid];
  /*reduce single-thread results within a block*/
  __shared__ double cache[1024];
  cache[threadIdx.x] = (*s);
  __syncthreads();
  int i;
  i = blockDim.x/2;
  while (i!=0)
  {
    if (threadIdx.x<i) 
      cache[threadIdx.x] = cache[threadIdx.x]+cache[threadIdx.x+i];
    __syncthreads();
    i = i/2;
  }
  if (threadIdx.x==0) 
    block_r[blockIdx.x] = cache[0];
}
