void VecAXPBYPCZ(int n, double a, double *x, double b, double *y, double c, double *z) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=16, cacheBlocks=True, pinHostMem=False, streamCount=2)
        for (i=0; i<=n-1; i++)
          y[i]=a*x[i]+b*y[i]+c*z[i];
    ) @*/
    {
      /*declare variables*/
      double *dev_a, *dev_c, *dev_b, *dev_y, *dev_x, *dev_z;
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
      cudaMalloc((void**)&dev_c,sizeof(double));
      cudaMalloc((void**)&dev_b,sizeof(double));
      cudaMalloc((void**)&dev_y,scSize);
      cudaHostRegister(y,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x,scSize);
      cudaHostRegister(x,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_z,scSize);
      cudaHostRegister(z,n,cudaHostRegisterPortable);
      /*copy data from host to device*/
      cudaMemcpy(dev_a,&a,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_c,&c,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_b,&b,sizeof(double),cudaMemcpyHostToDevice);
      int chunklen=n/nstreams;
      int chunkrem=n%nstreams;
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(dev_y+orcu_soff,y+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
        cudaMemcpyAsync(dev_x+orcu_soff,x+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
        cudaMemcpyAsync(dev_z+orcu_soff,z+orcu_soff,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        cudaMemcpyAsync(dev_z+orcu_soff,z+orcu_soff,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[orcu_i]);
      }
      /*invoke device kernel*/
      int blks4chunk=dimGrid.x/nstreams;
      if (dimGrid.x%nstreams!=0) 
        blks4chunk++ ;
      for (orcu_i=0; orcu_i<nstreams; orcu_i++ ) {
        orcu_soff=orcu_i*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[orcu_i]>>>(chunklen,dev_a,dev_c,dev_b,dev_y+orcu_soff,dev_x+orcu_soff,dev_z+orcu_soff);
      }
      if (chunkrem!=0) {
        orcu_soff=orcu_i*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[orcu_i]>>>(chunkrem,dev_a,dev_c,dev_b,dev_y+orcu_soff,dev_x+orcu_soff,dev_z+orcu_soff);
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
      cudaHostUnregister(y);
      cudaHostUnregister(x);
      cudaHostUnregister(z);
      for (orcu_i=0; orcu_i<=nstreams; orcu_i++ ) 
        cudaStreamDestroy(stream[orcu_i]);
      cudaFree(dev_a);
      cudaFree(dev_c);
      cudaFree(dev_b);
      cudaFree(dev_y);
      cudaFree(dev_x);
      cudaFree(dev_z);
    }
/*@ end @*/
}
__global__ void orcu_kernel3(int n, double* a, double* c, double* b, double* y, double* x, double* z) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double shared_y[16];
  __shared__ double shared_x[16];
  __shared__ double shared_z[16];
  if (tid<=n-1) {
    shared_y[threadIdx.x]=y[tid];
    shared_x[threadIdx.x]=x[tid];
    shared_z[threadIdx.x]=z[tid];
    shared_y[threadIdx.x]=(*a)*shared_x[threadIdx.x]+(*b)*shared_y[threadIdx.x]+(*c)*shared_z[threadIdx.x];
    y[tid]=shared_y[threadIdx.x];
  }
}
