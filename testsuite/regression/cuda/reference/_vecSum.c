void VecSum(int n, double *x, double s) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=16, cacheBlocks=True, pinHostMem=False, streamCount=2)
        for (i=0; i<=n-1; i++)
          s=s+x[i];
    ) @*/
    {
      /*declare variables*/
      double *dev_x, *dev_s;
      int nthreads=16;
      int nstreams=2;
      /*calculate device dimensions*/
      dim3 dimGrid, dimBlock;
      dimBlock.x=nthreads;
      dimGrid.x=(n+nthreads-1)/nthreads;
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
      cudaMalloc((void**)&dev_s,(dimGrid.x+1)*sizeof(double));
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
        orcu_kernel4<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,dev_x+soffset,dev_s+boffset);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        boffset=istream*blks4chunk;
        orcu_kernel4<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,dev_x+soffset,dev_s+boffset);
        blks4chunks++ ;
      }
      int orcu_blks=blks4chunks;
      int orcu_trds, orcu_n;
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
        orcu_blksum5<<<orcu_blks,orcu_trds>>>(orcu_n,dev_s);
      }
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&orcu_elapsed,start,stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      /*copy data from device to host*/
      cudaMemcpy(&s,dev_s,sizeof(double),cudaMemcpyDeviceToHost);
      cudaHostUnregister(x);
      for (istream=0; istream<=nstreams; istream++ ) 
        cudaStreamDestroy(stream[istream]);
      /*free allocated memory*/
      cudaFree(dev_x);
      cudaFree(dev_s);
    }
/*@ end @*/
}
__global__ void orcu_kernel4(int n, double* x, double* reducts) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double shared_x[16];
  double orcu_var1=0;
  if (tid<=n-1) {
    shared_x[threadIdx.x]=x[tid];
    orcu_var1=orcu_var1+shared_x[threadIdx.x];
  }
  /*reduce single-thread results within a block*/
  __shared__ double orcu_vec2[16];
  orcu_vec2[threadIdx.x]=orcu_var1;
  __shared__ double orcu_acc;
  if (threadIdx.x==0) 
    orcu_acc=0;
  __syncthreads();
  int orcu_i;
  int orcu_rem=0;
  for (orcu_i=blockDim.x/2; orcu_i>0; orcu_i>>=1) {
    if (threadIdx.x<orcu_i) 
      orcu_vec2[threadIdx.x]+=orcu_vec2[threadIdx.x+orcu_i];
    else 
      if (orcu_rem&&orcu_rem==threadIdx.x) {
        orcu_acc+=orcu_vec2[orcu_rem-1];
        orcu_rem=0;
      } else 
        if (orcu_i&1&&orcu_i!=1&&threadIdx.x==orcu_i) 
          orcu_rem=orcu_i;
    __syncthreads();
  }
  if (threadIdx.x==0) 
    reducts[blockIdx.x]=orcu_vec2[0]+orcu_acc;
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
