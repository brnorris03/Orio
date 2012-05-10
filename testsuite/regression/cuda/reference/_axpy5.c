void axpy5(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4, double a5, double *x5) {
    
  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for (i=0; i<=n-1; i++)
    y[i]+=a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];

  ) @*/
  {
    /*declare variables*/
    double *dev_y, *dev_x2, *dev_x3, *dev_x1, *dev_x4, *dev_x5;
    int nthreads=32;
    int nstreams=2;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=14;
    /*create streams*/
    int istream, soffset;
    cudaStream_t stream[nstreams+1];
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamCreate(&stream[istream]);
    int chunklen=n/nstreams;
    int chunkrem=n%nstreams;
    /*allocate device memory*/
    int nbytes=n*sizeof(double);
    cudaMalloc((void**)&dev_y,nbytes);
    cudaHostRegister(y,nbytes,cudaHostRegisterPortable);
    cudaMalloc((void**)&dev_x2,nbytes);
    cudaHostRegister(x2,nbytes,cudaHostRegisterPortable);
    cudaMalloc((void**)&dev_x3,nbytes);
    cudaHostRegister(x3,nbytes,cudaHostRegisterPortable);
    cudaMalloc((void**)&dev_x1,nbytes);
    cudaHostRegister(x1,nbytes,cudaHostRegisterPortable);
    cudaMalloc((void**)&dev_x4,nbytes);
    cudaHostRegister(x4,nbytes,cudaHostRegisterPortable);
    cudaMalloc((void**)&dev_x5,nbytes);
    cudaHostRegister(x5,nbytes,cudaHostRegisterPortable);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    /*copy data from host to device*/
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    int blks4chunk=dimGrid.x/nstreams;
    if (dimGrid.x%nstreams!=0) 
      blks4chunk++ ;
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      orcu_kernel2<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,a1,a3,a2,a5,a4,dev_y+soffset,dev_x2+soffset,dev_x3+soffset,dev_x1+soffset,dev_x4+soffset,dev_x5+soffset);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      orcu_kernel2<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,a1,a3,a2,a5,a4,dev_y+soffset,dev_x2+soffset,dev_x3+soffset,dev_x1+soffset,dev_x4+soffset,dev_x5+soffset);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
    cudaHostUnregister(y);
    cudaHostUnregister(x2);
    cudaHostUnregister(x3);
    cudaHostUnregister(x1);
    cudaHostUnregister(x4);
    cudaHostUnregister(x5);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamDestroy(stream[istream]);
    /*free allocated memory*/
    cudaFree(dev_y);
    cudaFree(dev_x2);
    cudaFree(dev_x3);
    cudaFree(dev_x1);
    cudaFree(dev_x4);
    cudaFree(dev_x5);
  }
/*@ end @*/
}
__global__ void orcu_kernel2(const int n, double a1, double a3, double a2, double a5, double a4, double* y, double* x2, double* x3, double* x1, double* x4, double* x5) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  __shared__ double shared_y[32];
  __shared__ double shared_x2[32];
  __shared__ double shared_x3[32];
  __shared__ double shared_x1[32];
  __shared__ double shared_x4[32];
  __shared__ double shared_x5[32];
  for (int i=tid; i<=n-1; i+=gsize) {
    shared_y[threadIdx.x]=y[tid];
    shared_x2[threadIdx.x]=x2[tid];
    shared_x3[threadIdx.x]=x3[tid];
    shared_x1[threadIdx.x]=x1[tid];
    shared_x4[threadIdx.x]=x4[tid];
    shared_x5[threadIdx.x]=x5[tid];
    shared_y[threadIdx.x]=shared_y[threadIdx.x]+a1*shared_x1[threadIdx.x]+a2*shared_x2[threadIdx.x]+a3*shared_x3[threadIdx.x]+a4*shared_x4[threadIdx.x]+a5*shared_x5[threadIdx.x];
    y[tid]=shared_y[threadIdx.x];
  }
}
