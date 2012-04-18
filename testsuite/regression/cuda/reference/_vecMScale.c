void VecScaleMult(int n, double a, double *x) {

    register int i;

    /*@ begin Loop (
          transform CUDA(threadCount=16, cacheBlocks=True, pinHostMem=False, streamCount=2)
        for (i=0; i<=n-1; i++)
          x[i]=a*x[i];
    ) @*/
    {
      /*declare variables*/
      double *dev_a, *dev_x;
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
      cudaMalloc((void**)&dev_a,sizeof(double));
      cudaMalloc((void**)&dev_x,nbytes);
      cudaHostRegister(x,n,cudaHostRegisterPortable);
      /*copy data from host to device*/
      cudaMemcpy(dev_a,&a,sizeof(double),cudaMemcpyHostToDevice);
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(dev_x+soffset,x+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(dev_x+soffset,x+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
      }
      /*invoke device kernel*/
      orio_t_start=getClock();
      int blks4chunk=dimGrid.x/nstreams;
      if (dimGrid.x%nstreams!=0) 
        blks4chunk++ ;
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,dev_a,dev_x+soffset);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,dev_a,dev_x+soffset);
      }
      /*copy data from device to host*/
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(x+soffset,dev_x+soffset,chunklen*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        cudaMemcpyAsync(x+soffset,dev_x+soffset,chunkrem*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
      }
      for (istream=0; istream<=nstreams; istream++ ) 
        cudaStreamSynchronize(stream[istream]);
      for (istream=0; istream<=nstreams; istream++ ) 
        cudaStreamDestroy(stream[istream]);
      /*free allocated memory*/
      cudaFree(dev_a);
      cudaFree(dev_x);
    }
/*@ end @*/
}
__global__ void orcu_kernel3(int n, double* a, double* x) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double shared_x[16];
  if (tid<=n-1) {
    shared_x[threadIdx.x]=x[tid];
    shared_x[threadIdx.x]=(*a)*shared_x[threadIdx.x];
    x[tid]=shared_x[threadIdx.x];
  }
}
