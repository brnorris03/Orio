void axpy1(int n, double *y, double a1, double *x1)
{
register int i;


/*@ begin Loop(
  transform Composite(
    cuda = (16,True, False, 1)
    ,scalarreplace = (False, 'int')
, unrolljam = (['i'], [2])
  )
   {
    for (i=0; i<=n-1; i++) {
    	y[i]=y[i]+a1*x1[i];
    }
    
   }


   
  
) @*/
{
  {
    int orio_lbound1=0;
    {
      /*declare variables*/
      double *dev_y, *dev_x1;
      int nthreads=16;
      /*calculate device dimensions*/
      dim3 dimGrid, dimBlock;
      dimBlock.x=nthreads;
      dimGrid.x=(n+nthreads-1)/nthreads;
      /*allocate device memory*/
      int nbytes=n*sizeof(double);
      cudaMalloc((void**)&dev_y,nbytes);
      cudaMalloc((void**)&dev_x1,nbytes);
      /*copy data from host to device*/
      cudaMemcpy(dev_y,y,nbytes,cudaMemcpyHostToDevice);
      cudaMemcpy(dev_x1,x1,nbytes,cudaMemcpyHostToDevice);
      /*invoke device kernel*/
      int orcu_var3=orio_lbound1;
      orio_t_start=getClock();
      orcu_kernel6<<<dimGrid,dimBlock>>>(n,orcu_var3,a1,dev_y,dev_x1);
      /*copy data from device to host*/
      cudaMemcpy(y,dev_y,nbytes,cudaMemcpyDeviceToHost);
      /*free allocated memory*/
      cudaFree(dev_y);
      cudaFree(dev_x1);
    }
    int orio_lbound2=n-((n-(0))%2);
    {
      /*declare variables*/
      double *dev_y, *dev_x1;
      int nthreads=16;
      /*calculate device dimensions*/
      dim3 dimGrid, dimBlock;
      dimBlock.x=nthreads;
      dimGrid.x=(n+nthreads-1)/nthreads;
      /*allocate device memory*/
      int nbytes=n*sizeof(double);
      cudaMalloc((void**)&dev_y,nbytes);
      cudaMalloc((void**)&dev_x1,nbytes);
      /*copy data from host to device*/
      cudaMemcpy(dev_y,y,nbytes,cudaMemcpyHostToDevice);
      cudaMemcpy(dev_x1,x1,nbytes,cudaMemcpyHostToDevice);
      /*invoke device kernel*/
      int orcu_var8=orio_lbound2;
      orio_t_start=getClock();
      orcu_kernel11<<<dimGrid,dimBlock>>>(n,orcu_var8,a1,dev_y,dev_x1);
      /*copy data from device to host*/
      cudaMemcpy(y,dev_y,nbytes,cudaMemcpyDeviceToHost);
      /*free allocated memory*/
      cudaFree(dev_y);
      cudaFree(dev_x1);
    }
  }
}
/*@ end @*/
}

__global__ void orcu_kernel6(int n, int orcu_var3, double a1, double* y, double* x1) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x+orcu_var3;
  __shared__ double shared_y[16];
  __shared__ double shared_x1[16];
  if (tid<=n-2) {
    shared_y[threadIdx.x]=y[tid];
    shared_x1[threadIdx.x]=x1[tid];
    {
      shared_y[threadIdx.x]=shared_y[threadIdx.x]+a1*shared_x1[threadIdx.x];
      shared_y[threadIdx.x]=shared_y[threadIdx.x]+a1*shared_x1[threadIdx.x];
    }
    y[tid]=shared_y[threadIdx.x];
  }
}
__global__ void orcu_kernel11(int n, int orcu_var8, double a1, double* y, double* x1) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x+orcu_var8;
  __shared__ double shared_y[16];
  __shared__ double shared_x1[16];
  if (tid<=n-1) {
    shared_y[threadIdx.x]=y[tid];
    shared_x1[threadIdx.x]=x1[tid];
    shared_y[threadIdx.x]=shared_y[threadIdx.x]+a1*shared_x1[threadIdx.x];
    y[tid]=shared_y[threadIdx.x];
  }
}
