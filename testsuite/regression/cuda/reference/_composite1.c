void axpy1(int n, double *y, double a1, double *x1)
{
register int i;


/*@ begin Loop(
  transform Composite(
    cuda = (16,False, False, 1)
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
      double *dev_y, *dev_a1, *dev_x1;
      int nthreads=16;
      /*calculate device dimensions*/
      dim3 dimGrid, dimBlock;
      dimBlock.x=nthreads;
      dimGrid.x=(n+nthreads-1)/nthreads;
      /*allocate device memory*/
      int nbytes=n*sizeof(double);
      cudaMalloc((void**)&dev_a1,sizeof(double));
      cudaMalloc((void**)&dev_y,nbytes);
      cudaMalloc((void**)&dev_x1,nbytes);
      /*copy data from host to device*/
      cudaMemcpy(dev_a1,&a1,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_y,y,nbytes,cudaMemcpyHostToDevice);
      cudaMemcpy(dev_x1,x1,nbytes,cudaMemcpyHostToDevice);
      /*invoke device kernel*/
      orio_t_start=getClock();
      orcu_kernel5<<<dimGrid,dimBlock>>>(n,orio_lbound1,dev_y,dev_a1,dev_x1);
      /*copy data from device to host*/  //Below you are duplicating work. Fix this!!
      cudaMemcpy(y,dev_y,nbytes,cudaMemcpyDeviceToHost);
      cudaMemcpy(y,dev_y,nbytes,cudaMemcpyDeviceToHost);
      /*free allocated memory*/
      cudaFree(dev_y);
      cudaFree(dev_a1);
      cudaFree(dev_x1);
    }
    int orio_lbound2=n-((n-(0))%2);
    {
      /*declare variables*/
      double *dev_y, *dev_a1, *dev_x1;
      int nthreads=16;
      /*calculate device dimensions*/
      dim3 dimGrid, dimBlock;
      dimBlock.x=nthreads;
      dimGrid.x=(n+nthreads-1)/nthreads;
      /*allocate device memory*/
      int nbytes=n*sizeof(double);
      cudaMalloc((void**)&dev_a1,sizeof(double));
      cudaMalloc((void**)&dev_y,nbytes);
      cudaMalloc((void**)&dev_x1,nbytes);
      /*copy data from host to device*/
      cudaMemcpy(dev_a1,&a1,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_y,y,nbytes,cudaMemcpyHostToDevice);
      cudaMemcpy(dev_x1,x1,nbytes,cudaMemcpyHostToDevice);
      /*invoke device kernel*/
      orio_t_start=getClock();
      orcu_kernel9<<<dimGrid,dimBlock>>>(n,orio_lbound2,dev_y,dev_a1,dev_x1);
      /*copy data from device to host*/
      cudaMemcpy(y,dev_y,nbytes,cudaMemcpyDeviceToHost);
      /*free allocated memory*/
      cudaFree(dev_y);
      cudaFree(dev_a1);
      cudaFree(dev_x1);
    }
  }
}
/*@ end @*/
}

__global__ void orcu_kernel5(int n, int orio_lbound1, double* y, double* a1, double* x1) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x+orio_lbound1;
  if (tid<=n-2) {
    {
      y[tid]=y[tid]+(*a1)*x1[tid];
      y[(tid+1)]=y[(tid+1)]+(*a1)*x1[(tid+1)];
    }
  }
}
__global__ void orcu_kernel9(int n, int orio_lbound2, double* y, double* a1, double* x1) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x+orio_lbound2;
  if (tid<=n-1) {
    y[tid]=y[tid]+(*a1)*x1[tid];
  }
}
