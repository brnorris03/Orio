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
      cudaMalloc((void**)&dev_c,sizeof(double));
      cudaMalloc((void**)&dev_b,sizeof(double));
      cudaMalloc((void**)&dev_y,nbytes);
      cudaHostRegister(y,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_x,nbytes);
      cudaHostRegister(x,n,cudaHostRegisterPortable);
      cudaMalloc((void**)&dev_z,nbytes);
      cudaHostRegister(z,n,cudaHostRegisterPortable);
      /*copy data from host to device*/
      cudaMemcpy(dev_a,&a,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_c,&c,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(dev_b,&b,sizeof(double),cudaMemcpyHostToDevice);
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
      /*invoke device kernel*/
      orio_t_start=getClock();
      int blks4chunk=dimGrid.x/nstreams;
      if (dimGrid.x%nstreams!=0) 
        blks4chunk++ ;
      for (istream=0; istream<nstreams; istream++ ) {
        soffset=istream*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,dev_a,dev_c,dev_b,dev_y+soffset,dev_x+soffset,dev_z+soffset);
      }
      if (chunkrem!=0) {
        soffset=istream*chunklen;
        orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,dev_a,dev_c,dev_b,dev_y+soffset,dev_x+soffset,dev_z+soffset);
      }
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
      for (istream=0; istream<=nstreams; istream++ ) 
        cudaStreamDestroy(stream[istream]);
      /*free allocated memory*/
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
