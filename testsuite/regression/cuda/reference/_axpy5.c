void axpy5(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4, double a5, double *x5) {
    
    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=16, cacheBlocks=True, pinHostMem=False, streamCount=2)
        for (i=0; i<=n-1; i++)
          y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
    ) @*/
    {
      /*declare variables*/
      double *dev_a1, *dev_x2, *dev_a3, *dev_a2, *dev_a5, *dev_a4, *dev_y, *dev_x3, *dev_x1, *dev_x4, *dev_x5;
      int nthreads=16;
      int nstreams=2;
      /*calculate device dimensions*/
      dim3 dimGrid, dimBlock;
      dimBlock.x=nthreads;
      dimGrid.x=(n+nthreads-1)/nthreads;
      /*create streams*/
      int istream, soffset;
      cudaStream_t stream[nstreams+1];
      for (istream=0; istream<=nstreams; istream++ ) 
        cudaStreamCreate(&stream[istream]);
      int chunklen=n/nstreams;
      int chunkrem=n%nstreams;
      /*allocate device memory*/
      cudaMalloc((void**)&dev_a1,sizeof(double));
      cudaMalloc((void**)&dev_a3,sizeof(double));
      cudaMalloc((void**)&dev_a2,sizeof(double));
      cudaMalloc((void**)&dev_a5,sizeof(double));
      cudaMalloc((void**)&dev_a4,sizeof(double));
      cudaMalloc((void**)&dev_y,sizeof(y));
      cudaHostRegister(y,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x2,sizeof(x2));
      cudaHostRegister(x2,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x3,sizeof(x3));
      cudaHostRegister(x3,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x1,sizeof(x1));
      cudaHostRegister(x1,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x4,sizeof(x4));
      cudaHostRegister(x4,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x5,sizeof(x5));
      cudaHostRegister(x5,n,cudaHostRegisterPortable);
      /*copy data from host to device*/
      cudaMemcpy(dev_a1,&a1,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a3,&a3,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a2,&a2,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a5,&a5,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a4,&a4,sizeof(double),cudaMemcpyHostToDevice);
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(dev_y+soffset,y+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x2+soffset,x2+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x3+soffset,x3+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x1+soffset,x1+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x4+soffset,x4+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x5+soffset,x5+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(dev_y+soffset,y+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x2+soffset,x2+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x3+soffset,x3+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x1+soffset,x1+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x4+soffset,x4+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x5+soffset,x5+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      }
      /*invoke device kernel*/
      orio_t_start=getClock();
      int blks4chunk=dimGrid.x/nstreams;
      if (dimGrid.x%nstreams!=0) 
        blks4chunk++ ;
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,dev_a1,dev_x2+soffset,dev_a3,dev_a2,dev_a5,dev_a4,dev_y+soffset,dev_x3+soffset,dev_x1+soffset,dev_x4+soffset,dev_x5+soffset);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,dev_a1,dev_x2+soffset,dev_a3,dev_a2,dev_a5,dev_a4,dev_y+soffset,dev_x3+soffset,dev_x1+soffset,dev_x4+soffset,dev_x5+soffset);
      }
      /*copy data from device to host*/
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(y+soffset,dev_y+soffset,chunklen*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(y+soffset,dev_y+soffset,chunkrem*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
      }
      for (istream=0; istream<=nstreams; istream++ ) 
        cudaStreamSynchronize(stream[istream]);
      for (istream=0; istream<=nstreams; istream++ ) 
        cudaStreamDestroy(stream[istream]);
      /*free allocated memory*/
      cudaFree(dev_a1);
      cudaFree(dev_x2);
      cudaFree(dev_a3);
      cudaFree(dev_a2);
      cudaFree(dev_a5);
      cudaFree(dev_a4);
      cudaFree(dev_y);
      cudaFree(dev_x3);
      cudaFree(dev_x1);
      cudaFree(dev_x4);
      cudaFree(dev_x5);
    }
/*@ end @*/
}
__global__ void orcu_kernel3(int n, double* a1, double* x2, double* a3, double* a2, double* a5, double* a4, double* y, double* x3, double* x1, double* x4, double* x5) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double shared_y[16];
  __shared__ double shared_x2[16];
  __shared__ double shared_x3[16];
  __shared__ double shared_x1[16];
  __shared__ double shared_x4[16];
  __shared__ double shared_x5[16];
  if (tid<=n-1) {
    shared_y[threadIdx.x]=y[tid];
    shared_x2[threadIdx.x]=x2[tid];
    shared_x3[threadIdx.x]=x3[tid];
    shared_x1[threadIdx.x]=x1[tid];
    shared_x4[threadIdx.x]=x4[tid];
    shared_x5[threadIdx.x]=x5[tid];
    shared_y[threadIdx.x]=shared_y[threadIdx.x]+(*a1)*shared_x1[threadIdx.x]+(*a2)*shared_x2[threadIdx.x]+(*a3)*shared_x3[threadIdx.x]+(*a4)*shared_x4[threadIdx.x]+(*a5)*shared_x5[threadIdx.x];
    y[tid]=shared_y[threadIdx.x];
  }
}
