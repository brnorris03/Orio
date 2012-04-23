void VecWAXPY(int n, double *w, double a, double *x, double *y) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=16, cacheBlocks=True, pinHostMem=False, streamCount=2)
        for (i=0; i<=n-1; i++)
          w[i]=a*x[i]+y[i];
    ) @*/
    {
      /*declare variables*/
      double *dev_x, *dev_y, *dev_w;
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
      int nbytes=n*sizeof(double);
      cudaMalloc((void**)&dev_y,nbytes);
      cudaHostRegister(y,nbytes,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x,nbytes);
      cudaHostRegister(x,nbytes,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_w,nbytes);
      /*copy data from host to device*/
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(dev_y+soffset,y+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x+soffset,x+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(dev_y+soffset,y+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
        cudaMemcpyAsync(dev_x+soffset,x+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      }
      /*invoke device kernel*/
      orio_t_start=getClock();
      int blks4chunk=dimGrid.x/nstreams;
      if (dimGrid.x%nstreams!=0) 
        blks4chunk++ ;
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,a,dev_x+soffset,dev_y+soffset,dev_w+soffset);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,a,dev_x+soffset,dev_y+soffset,dev_w+soffset);
      }
      /*copy data from device to host*/
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(w+soffset,dev_w+soffset,chunklen*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(w+soffset,dev_w+soffset,chunkrem*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
      }
      for (istream=0; istream<=nstreams; istream++ ) 
        cudaStreamSynchronize(stream[istream]);
      cudaHostUnregister(y);
      cudaHostUnregister(x);
      for (istream=0; istream<=nstreams; istream++ ) 
        cudaStreamDestroy(stream[istream]);
      /*free allocated memory*/
      cudaFree(dev_x);
      cudaFree(dev_y);
      cudaFree(dev_w);
    }
/*@ end @*/
}
__global__ void orcu_kernel3(int n, double a, double* x, double* y, double* w) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double shared_y[16];
  __shared__ double shared_x[16];
  __shared__ double shared_w[16];
  if (tid<=n-1) {
    shared_y[threadIdx.x]=y[tid];
    shared_x[threadIdx.x]=x[tid];
    shared_w[threadIdx.x]=a*shared_x[threadIdx.x]+shared_y[threadIdx.x];
    w[tid]=shared_w[threadIdx.x];
  }
}
