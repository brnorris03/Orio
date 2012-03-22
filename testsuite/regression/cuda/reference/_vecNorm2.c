void VecNorm2(int n, double *x, double r) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=16, cacheBlocks=True, pinHostMem=False, streamCount=2)
        for (i=0; i<=n-1; i++)
          r=r+x[i]*x[i];
        r=sqrt(r);
    ) @*/
    {
      /*declare variables*/
      double *dev_x, *dev_reducts;
      int nthreads=16;
      int nstreams=2;
      dim3 dimGrid, dimBlock;
      int orcu_i;
      int orcu_n;
      cudaStream_t stream[nstreams+1];
      int orcu_soff;
      int orcu_boff;
      /*calculate device dimensions*/
      dimGrid.x=ceil((float)n/(float)nthreads);
      dimBlock.x=nthreads;
      /*allocate device memory*/
      int scSize=n*sizeof(double);
      for (orcu_i=0; orcu_i<=nstreams; orcu_i++ ) 
        cudaStreamCreate(&stream[orcu_i]);
      cudaMalloc((void**)&dev_x,scSize);
      cudaHostRegister(x,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_reducts,(dimGrid.x+1)*sizeof(double));
      /*copy data from host to device*/
      int chunklen=n/nstreams;
      int chunkrem=n%nstreams;
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(dev_x+orcu_soff,x+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(dev_x+orcu_soff,x+orcu_soff,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
      }
      /*invoke device kernel*/
      int blks4chunk=dimGrid.x/nstreams;
      if (dimGrid.x%nstreams!=0) 
        blks4chunk++ ;
      int blks4chunks=blks4chunk*nstreams;
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        orcu_boff=orcu_i*blks4chunk;
        orcu_kernel4<<<blks4chunk,dimBlock,0,stream[orcu_i]>>>(chunklen,dev_x+orcu_soff,dev_reducts+orcu_boff);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        orcu_boff=orcu_i*blks4chunk;
        orcu_kernel4<<<blks4chunk,dimBlock,0,stream[orcu_i]>>>(chunkrem,dev_x+orcu_soff,dev_reducts+orcu_boff);
        blks4chunks++ ;
      }
      int orcu_blks=blks4chunks;
      int orcu_trds;
      while (orcu_blks>1) {
        cudaDeviceSynchronize();
        orcu_n=orcu_blks;
        orcu_blks=(orcu_blks+1023)/1024;
        if (orcu_n<1024) {
          orcu_trds=1;
          while (orcu_trds<orcu_n) 
            orcu_trds<<=1;
        } else 
          orcu_trds=1024;
        orcu_blksum5<<<orcu_blks,orcu_trds>>>(orcu_n,dev_reducts);
      }
      /*copy data from device to host*/
      cudaMemcpy(&r,dev_reducts,sizeof(double),cudaMemcpyDeviceToHost);
      /*free allocated memory*/
      cudaHostUnregister(x);
      for (orcu_i=0; orcu_i<=nstreams; orcu_i++ ) 
        cudaStreamDestroy(stream[orcu_i]);
      cudaFree(dev_x);
      cudaFree(dev_reducts);
    }
    r=sqrt(r);
/*@ end @*/
}
__global__ void orcu_kernel4(int n, double* x, double* reducts) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double shared_x[16];
  double orcu_var1=0;
  if (tid<=n-1) {
    shared_x[threadIdx.x]=x[tid];
    orcu_var1=orcu_var1+shared_x[threadIdx.x]*shared_x[threadIdx.x];
  }
  /*reduce single-thread results within a block*/
  __shared__ double orcu_vec2[16];
  orcu_vec2[threadIdx.x]=orcu_var1;
  __syncthreads();
  int orcu_i;
  for (orcu_i=blockDim.x/2; orcu_i>0; orcu_i>>=1) {
    if (threadIdx.x<orcu_i) 
      orcu_vec2[threadIdx.x]+=orcu_vec2[threadIdx.x+orcu_i];
    __syncthreads();
  }
  if (threadIdx.x==0) 
    reducts[blockIdx.x]=orcu_vec2[0];
}

__global__ void orcu_blksum5(int orcu_n, double* reducts) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double orcu_vec3[1024];
  if (tid<orcu_n) 
    orcu_vec3[threadIdx.x]=reducts[tid];
  else 
    orcu_vec3[threadIdx.x]=0;
  __syncthreads();
  int orcu_i;
  for (orcu_i=blockDim.x/2; orcu_i>0; orcu_i>>=1) {
    if (threadIdx.x<orcu_i) 
      orcu_vec3[threadIdx.x]+=orcu_vec3[threadIdx.x+orcu_i];
    __syncthreads();
  }
  if (threadIdx.x==0) 
    reducts[blockIdx.x]=orcu_vec3[0];
}
