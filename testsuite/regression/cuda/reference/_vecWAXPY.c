void VecWAXPY(int n, double *w, double a, double *x, double *y) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=16, cacheBlocks=True, pinHostMem=False, streamCount=2)
        for (i=0; i<=n-1; i++)
          w[i]=a*x[i]+y[i];
    ) @*/
    {
      /*declare variables*/
      double *dev_a, *dev_x, *dev_y, *dev_w;
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
      cudaMalloc((void**)&dev_a,sizeof(double));
      cudaMalloc((void**)&dev_y,scSize);
      cudaHostRegister(y,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x,scSize);
      cudaHostRegister(x,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_w,scSize);
      /*copy data from host to device*/
      cudaMemcpy(dev_a,&a,sizeof(double),cudaMemcpyHostToDevice);
      int chunklen=n/nstreams;
      int chunkrem=n%nstreams;
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(dev_y+orcu_soff,y+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
        cudaMemcpyAsync(dev_x+orcu_soff,x+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(dev_w+orcu_soff,w+orcu_soff,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
      }
      /*invoke device kernel*/
      int blks4chunk=dimGrid.x/nstreams;
      if (dimGrid.x%nstreams!=0) 
        blks4chunk++ ;
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[orcu_i]>>>(chunklen,dev_a,dev_x+orcu_soff,dev_y+orcu_soff,dev_w+orcu_soff);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[orcu_i]>>>(chunkrem,dev_a,dev_x+orcu_soff,dev_y+orcu_soff,dev_w+orcu_soff);
      }
      /*copy data from device to host*/
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(w+orcu_soff,dev_w+orcu_soff,chunklen*sizeof(double),cudaMemcpyDeviceToHost,stream[orcu_i]);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(w+orcu_soff,dev_w+orcu_soff,chunkrem*sizeof(double),cudaMemcpyDeviceToHost,stream[orcu_i]);
      }
      for (orcu_i=0; orcu_i<=nstreams; orcu_i++ ) 
        cudaStreamSynchronize(stream[orcu_i]);
      /*free allocated memory*/
      cudaHostUnregister(y);
      cudaHostUnregister(x);
      for (orcu_i=0; orcu_i<=nstreams; orcu_i++ ) 
        cudaStreamDestroy(stream[orcu_i]);
      cudaFree(dev_a);
      cudaFree(dev_x);
      cudaFree(dev_y);
      cudaFree(dev_w);
    }
/*@ end @*/
}
__global__ void orcu_kernel3(int n, double* a, double* x, double* y, double* w) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double shared_y[16];
  __shared__ double shared_x[16];
  __shared__ double shared_w[16];
  if (tid<=n-1) {
    shared_y[threadIdx.x]=y[tid];
    shared_x[threadIdx.x]=x[tid];
    shared_w[threadIdx.x]=(*a)*shared_x[threadIdx.x]+shared_y[threadIdx.x];
    w[tid]=shared_w[threadIdx.x];
  }
}
