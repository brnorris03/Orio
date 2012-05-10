void VecAXPBYPCZ(int n, double a, double *x, double b, double *y, double c, double *z) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for (i=0; i<=n-1; i++)
    y[i]=a*x[i]+b*y[i]+c*z[i];

  ) @*/
  {
    /*declare variables*/
    double *dev_y, *dev_x, *dev_z;
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
    cudaMalloc((void**)&dev_x,nbytes);
    cudaHostRegister(x,nbytes,cudaHostRegisterPortable);
    cudaMalloc((void**)&dev_z,nbytes);
    cudaHostRegister(z,nbytes,cudaHostRegisterPortable);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    /*copy data from host to device*/
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_y+soffset,y+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      cudaMemcpyAsync(dev_x+soffset,x+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      cudaMemcpyAsync(dev_z+soffset,z+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_y+soffset,y+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      cudaMemcpyAsync(dev_x+soffset,x+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      cudaMemcpyAsync(dev_z+soffset,z+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
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
      orcu_kernel2<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,a,c,b,dev_y+soffset,dev_x+soffset,dev_z+soffset);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      orcu_kernel2<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,a,c,b,dev_y+soffset,dev_x+soffset,dev_z+soffset);
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
    cudaHostUnregister(x);
    cudaHostUnregister(z);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamDestroy(stream[istream]);
    /*free allocated memory*/
    cudaFree(dev_y);
    cudaFree(dev_x);
    cudaFree(dev_z);
  }
/*@ end @*/
}
__global__ void orcu_kernel2(const int n, double a, double c, double b, double* y, double* x, double* z) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  __shared__ double shared_y[32];
  __shared__ double shared_x[32];
  __shared__ double shared_z[32];
  for (int i=tid; i<=n-1; i+=gsize) {
    shared_y[threadIdx.x]=y[tid];
    shared_x[threadIdx.x]=x[tid];
    shared_z[threadIdx.x]=z[tid];
    shared_y[threadIdx.x]=a*shared_x[threadIdx.x]+b*shared_y[threadIdx.x]+c*shared_z[threadIdx.x];
    y[tid]=shared_y[threadIdx.x];
  }
}
