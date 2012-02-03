void VecScaleMult(int n, double a, double *x) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=1024, maxBlocks=65535)
        for (i=0; i<=n-1; i++)
          x[i]=a*x[i];
    ) @*/
    {
      /*declare device variables*/
      double *dev_a, *dev_x;
      int* dev_n;
      dim3 dimGrid, dimBlock;
      /*calculate device dimensions*/
      dimGrid.x = ceil((float)n/(float)1024);
      dimBlock.x = 1024;
      /*allocate device memory*/
      cudaMalloc((void**)&dev_n,sizeof(int));
      cudaMalloc((void**)&dev_a,sizeof(double));
      cudaMalloc((void**)&dev_x,n*sizeof(double));
      /*copy data from host to devices*/
      cudaMemcpy(dev_n,&n,sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a,&a,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_x,x,n*sizeof(double),cudaMemcpyHostToDevice);
      /*invoke device kernel function*/
      orcuda_kern_1<<<dimGrid,dimBlock>>>(dev_n,dev_a,dev_x);
      /*copy data from devices to host*/
      cudaMemcpy(x,dev_x,n*sizeof(double),cudaMemcpyDeviceToHost);
      /*free device memory*/
      cudaFree(dev_n);
      cudaFree(dev_a);
      cudaFree(dev_x);
    }
/*@ end @*/
}
__global__ void orcuda_kern_1(int* n, double* a, double* x) {
  int tid;
  tid = blockIdx.x*blockDim.x+threadIdx.x;
  if (tid<=(*n)-1) 
    x[tid]=(*a)*x[tid];
}
