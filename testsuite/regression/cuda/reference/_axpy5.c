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
      dim3 dimGrid, dimBlock;
      int orcu_i;
      cudaStream_t stream[nstreams+1];
      int orcu_soff;
      /*calculate device dimensions*/
      dimGrid.x=ceil((float)n/(float)nthreads);
      dimBlock.x=nthreads;
      /*allocate device memory*/
      int scSize=n*sizeof(double);
      for (orcu_i=0; orcu_i<=nstreams; orcu_i++ ) 
        cudaStreamCreate(&stream[orcu_i]);
      cudaMalloc((void**)&dev_a1,sizeof(double));
      cudaMalloc((void**)&dev_a3,sizeof(double));
      cudaMalloc((void**)&dev_a2,sizeof(double));
      cudaMalloc((void**)&dev_a5,sizeof(double));
      cudaMalloc((void**)&dev_a4,sizeof(double));
      cudaMalloc((void**)&dev_x2,scSize);
      cudaHostRegister(x2,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_y,scSize);
      cudaHostRegister(y,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x3,scSize);
      cudaHostRegister(x3,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x1,scSize);
      cudaHostRegister(x1,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x4,scSize);
      cudaHostRegister(x4,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x5,scSize);
      cudaHostRegister(x5,n,cudaHostRegisterPortable);
      /*copy data from host to device*/
      cudaMemcpy(dev_a1,&a1,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a3,&a3,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a2,&a2,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a5,&a5,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_a4,&a4,sizeof(double),cudaMemcpyHostToDevice);
      int chunklen=n/nstreams;
      int chunkrem=n%nstreams;
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(dev_x2+orcu_soff,x2+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
        cudaMemcpyAsync(dev_y+orcu_soff,y+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
        cudaMemcpyAsync(dev_x3+orcu_soff,x3+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
        cudaMemcpyAsync(dev_x1+orcu_soff,x1+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
        cudaMemcpyAsync(dev_x4+orcu_soff,x4+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
        cudaMemcpyAsync(dev_x5+orcu_soff,x5+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(dev_x5+orcu_soff,x5+orcu_soff,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
      }
      /*invoke device kernel*/
      int blks4chunk=dimGrid.x/nstreams;
      if (dimGrid.x%nstreams!=0) 
        blks4chunk++ ;
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[orcu_i]>>>(chunklen,dev_a1,dev_x2+orcu_soff,dev_a3,dev_a2,dev_a5,dev_a4,dev_y+orcu_soff,dev_x3+orcu_soff,dev_x1+orcu_soff,dev_x4+orcu_soff,dev_x5+orcu_soff);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[orcu_i]>>>(chunkrem,dev_a1,dev_x2+orcu_soff,dev_a3,dev_a2,dev_a5,dev_a4,dev_y+orcu_soff,dev_x3+orcu_soff,dev_x1+orcu_soff,dev_x4+orcu_soff,dev_x5+orcu_soff);
      }
      /*copy data from device to host*/
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(y+orcu_soff,dev_y+orcu_soff,chunklen*sizeof(double),cudaMemcpyDeviceToHost,stream[orcu_i]);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(y+orcu_soff,dev_y+orcu_soff,chunkrem*sizeof(double),cudaMemcpyDeviceToHost,stream[orcu_i]);
      }
      for (orcu_i=0; orcu_i<=nstreams; orcu_i++ ) 
        cudaStreamSynchronize(stream[orcu_i]);
      /*free allocated memory*/
      cudaHostUnregister(x2);
      cudaHostUnregister(y);
      cudaHostUnregister(x3);
      cudaHostUnregister(x1);
      cudaHostUnregister(x4);
      cudaHostUnregister(x5);
      for (orcu_i=0; orcu_i<=nstreams; orcu_i++ ) 
        cudaStreamDestroy(stream[orcu_i]);
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
  __shared__ double shared_x2[16];
  __shared__ double shared_y[16];
  __shared__ double shared_x3[16];
  __shared__ double shared_x1[16];
  __shared__ double shared_x4[16];
  __shared__ double shared_x5[16];
  if (tid<=n-1) {
    shared_x2[threadIdx.x]=x2[tid];
    shared_y[threadIdx.x]=y[tid];
    shared_x3[threadIdx.x]=x3[tid];
    shared_x1[threadIdx.x]=x1[tid];
    shared_x4[threadIdx.x]=x4[tid];
    shared_x5[threadIdx.x]=x5[tid];
    shared_y[threadIdx.x]=shared_y[threadIdx.x]+(*a1)*shared_x1[threadIdx.x]+(*a2)*shared_x2[threadIdx.x]+(*a3)*shared_x3[threadIdx.x]+(*a4)*shared_x4[threadIdx.x]+(*a5)*shared_x5[threadIdx.x];
    y[tid]=shared_y[threadIdx.x];
  }
}
