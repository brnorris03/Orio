void VecAXPBY(int n, double a, double *x, double b, double *y) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=1024, maxBlocks=65535)
        for (i=0; i<=n-1; i++)
          y[i]=a*x[i]+b*y[i];
    ) @*/
    {
      /*declare device variables*/
      double *dev_y, *dev_a, *dev_b, *dev_x;
      int* dev_n;
      dim3 dimGrid, dimBlock;
      /*calculate device dimensions*/
      dimGrid.x = ceil((float)n/(float)1024);
      dimBlock.x = 1024;
      /*allocate device memory*/
      cudaMalloc((void**)&dev_n,sizeof(int));
      cudaMalloc((void**)&dev_a,sizeof(double));
      cudaMalloc((void**)&dev_b,sizeof(double));
      cudaMalloc((void**)&dev_y,n*sizeof(double));
      cudaMalloc((void**)&dev_x,n*sizeof(double));
      /*copy data from host to devices*/
      cudaMemcpy(dev_n,&n,sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a,&a,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_b,&b,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_y,y,n*sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_x,x,n*sizeof(double),cudaMemcpyHostToDevice);
      /*invoke device kernel function*/
      orcuda_kern_1<<<dimGrid,dimBlock>>>(dev_n,dev_y,dev_a,dev_b,dev_x);
      /*copy data from devices to host*/
      cudaMemcpy(y,dev_y,n*sizeof(double),cudaMemcpyDeviceToHost);
      /*free device memory*/
      cudaFree(dev_n);
      cudaFree(dev_y);
      cudaFree(dev_a);
      cudaFree(dev_b);
      cudaFree(dev_x);
    }
/*@ end @*/
}
__global__ void orcuda_kern_1(int* n, double* y, double* a, double* b, double* x) {
  int tid;
  tid = blockIdx.x*blockDim.x+threadIdx.x;
  if (tid<=(*n)-1) 
    y[tid]=(*a)*x[tid]+(*b)*y[tid];
}
