void VecNorm2(int n, double *x, double r) {

  register int i;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, cacheBlocks=True, preferL1Size=16)

  for (i=0; i<=n-1; i++)
    r+=x[i]*x[i];
  r=sqrt(r);

  ) @*/
  {
    /*declare variables*/
    double *dev_x, *dev_r;
    int nthreads=32;
    int nstreams=2;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=14;
    /*create streams*/
    int istream, soffset, boffset;
    cudaStream_t stream[nstreams+1];
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamCreate(&stream[istream]);
    int chunklen=n/nstreams;
    int chunkrem=n%nstreams;
    /*allocate device memory*/
    int nbytes=n*sizeof(double);
    cudaMalloc((void**)&dev_x,nbytes);
    cudaHostRegister(x,nbytes,cudaHostRegisterPortable);
    cudaMalloc((void**)&dev_r,(dimGrid.x+1)*sizeof(double));
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    /*copy data from host to device*/
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_x+soffset,x+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_x+soffset,x+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    int blks4chunk=dimGrid.x/nstreams;
    if (dimGrid.x%nstreams!=0) 
      blks4chunk++ ;
    int blks4chunks=blks4chunk*nstreams;
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      boffset=istream*blks4chunk;
      orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,dev_x+soffset,dev_r+boffset);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      boffset=istream*blks4chunk;
      orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,dev_x+soffset,dev_r+boffset);
      blks4chunks++ ;
    }
    int orcu_blks=blks4chunks;
    int orcu_n;
    while (orcu_blks>1) {
      orcu_n=orcu_blks;
      orcu_blks=(orcu_blks+31)/32;
      orcu_blksum4<<<orcu_blks,32>>>(orcu_n,dev_r);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /*copy data from device to host*/
    cudaMemcpy(&r,dev_r,sizeof(double),cudaMemcpyDeviceToHost);
    cudaHostUnregister(x);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamDestroy(stream[istream]);
    /*free allocated memory*/
    cudaFree(dev_x);
    cudaFree(dev_r);
  }
  r=sqrt(r);
/*@ end @*/
}
__device__ void orcu_warpReduce32(int tid, volatile double* reducts) {
  reducts[tid]+=reducts[tid+16];
  reducts[tid]+=reducts[tid+8];
  reducts[tid]+=reducts[tid+4];
  reducts[tid]+=reducts[tid+2];
  reducts[tid]+=reducts[tid+1];
}

__device__ void orcu_warpReduce64(int tid, volatile double* reducts) {
  reducts[tid]+=reducts[tid+32];
  reducts[tid]+=reducts[tid+16];
  reducts[tid]+=reducts[tid+8];
  reducts[tid]+=reducts[tid+4];
  reducts[tid]+=reducts[tid+2];
  reducts[tid]+=reducts[tid+1];
}

__global__ void orcu_kernel3(const int n, double* x, double* reducts) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  __shared__ double shared_x[32];
  double orcu_var1=0;
  for (int i=tid; i<=n-1; i+=gsize) {
    shared_x[threadIdx.x]=x[tid];
    orcu_var1=orcu_var1+shared_x[threadIdx.x]*shared_x[threadIdx.x];
  }
  /*reduce single-thread results within a block*/
  __shared__ double orcu_vec2[32];
  orcu_vec2[threadIdx.x]=orcu_var1;
  if (threadIdx.x<16) 
    orcu_warpReduce32(threadIdx.x,orcu_vec2);
  __syncthreads();
  if (threadIdx.x==0) 
    reducts[blockIdx.x]=orcu_vec2[0];
}

__global__ void orcu_blksum4(int orcu_n, double* reducts) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double orcu_vec2[32];
  if (tid<orcu_n) 
    orcu_vec2[threadIdx.x]=reducts[tid];
  else 
    orcu_vec2[threadIdx.x]=0;
  if (threadIdx.x<16) 
    orcu_warpReduce32(threadIdx.x,orcu_vec2);
  __syncthreads();
  if (threadIdx.x==0) 
    reducts[blockIdx.x]=orcu_vec2[0];
}
