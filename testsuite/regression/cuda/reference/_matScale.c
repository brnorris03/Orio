void MatScale_SeqDIA(double* A, double a) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for(i=0;i<=nz-1;i++) {
    A[i] *= a;
  }

  ) @*/
  {
    /*declare variables*/
    double *dev_A;
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
    int chunklen=nz/nstreams;
    int chunkrem=nz%nstreams;
    /*allocate device memory*/
    int nbytes=nz*sizeof(double);
    cudaMalloc((void**)&dev_A,nbytes);
    cudaHostRegister(A,nbytes,cudaHostRegisterPortable);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    /*copy data from host to device*/
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_A+soffset,A+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_A+soffset,A+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
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
      orcu_kernel2<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,a,dev_A+soffset);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      orcu_kernel2<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,a,dev_A+soffset);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /*copy data from device to host*/
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(A+soffset,dev_A+soffset,chunklen*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(A+soffset,dev_A+soffset,chunkrem*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
    }
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamSynchronize(stream[istream]);
    cudaHostUnregister(A);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamDestroy(stream[istream]);
    /*free allocated memory*/
    cudaFree(dev_A);
  }
/*@ end @*/
}
__global__ void orcu_kernel2(const int nz, double a, double* A) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  __shared__ double shared_A[32];
  for (int i=tid; i<=nz-1; i+=gsize) {
    shared_A[threadIdx.x]=A[tid];
    shared_A[threadIdx.x]=shared_A[threadIdx.x]*a;
    A[tid]=shared_A[threadIdx.x];
  }
}
