void FormFunction2D(double lambda, int m, int n, double* X, double *F) {
  int i;
  int nrows=m*n;
  int offsets[5];
  offsets[0]=-m;
  offsets[1]=-1;
  offsets[2]=0;
  offsets[3]=1;
  offsets[4]=m;
  int bb = offsets[4];
  int be = nrows-offsets[4];

  double hx     = 1.0/(m-1);
  double hy     = 1.0/(n-1);
  double sc     = hx*hy*lambda;
  double hxdhy  = hx/hy;
  double hydhx  = hy/hx;
  double u;

  /*@ begin Loop(transform CUDA(threadCount=32, blockCount=14, streamCount=2, preferL1Size=16)

  for(i=bb; i<=be-1; i++) {
    F[i] = (2*X[i+2*nrows] - X[i+nrows] - X[i+3*nrows])*hydhx + (2*X[i+2*nrows] - X[i] - X[i+4*nrows])*hxdhy - sc*exp(X[i+2*nrows]);
  }

  ) @*/
  {
    /*declare variables*/
    double *dev_F, *dev_X;
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
    int chunklen=be/nstreams;
    int chunkrem=be%nstreams;
    /*allocate device memory*/
    int nbytes=be*sizeof(double);
    cudaMalloc((void**)&dev_X,nbytes);
    cudaHostRegister(X,nbytes,cudaHostRegisterPortable);
    cudaMalloc((void**)&dev_F,nbytes);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    /*copy data from host to device*/
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_X+soffset,X+soffset,chunklen*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(dev_X+soffset,X+soffset,chunkrem*sizeof(double),cudaMemcpyHostToDevice,stream[istream]);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    int orcu_var1=bb;
    int blks4chunk=dimGrid.x/nstreams;
    if (dimGrid.x%nstreams!=0) 
      blks4chunk++ ;
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunklen,nrows,orcu_var1,hxdhy,sc,hydhx,dev_F+soffset,dev_X+soffset);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      orcu_kernel3<<<blks4chunk,dimBlock,0,stream[istream]>>>(chunkrem,nrows,orcu_var1,hxdhy,sc,hydhx,dev_F+soffset,dev_X+soffset);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /*copy data from device to host*/
    for (istream=0; istream<nstreams; istream++ ) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(F+soffset,dev_F+soffset,chunklen*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
    }
    if (chunkrem!=0) {
      soffset=istream*chunklen;
      cudaMemcpyAsync(F+soffset,dev_F+soffset,chunkrem*sizeof(double),cudaMemcpyDeviceToHost,stream[istream]);
    }
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamSynchronize(stream[istream]);
    cudaHostUnregister(X);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    for (istream=0; istream<=nstreams; istream++ ) 
      cudaStreamDestroy(stream[istream]);
    /*free allocated memory*/
    cudaFree(dev_F);
    cudaFree(dev_X);
  }
/*@ end @*/
}
__global__ void orcu_kernel3(const int be, int nrows, int orcu_var1, double hxdhy, double sc, double hydhx, double* F, double* X) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x+orcu_var1;
  const int gsize=gridDim.x*blockDim.x;
  for (int i=tid; i<=be-1; i+=gsize) {
    F[i]=(2*X[i+2*nrows]-X[i+nrows]-X[i+3*nrows])*hydhx+(2*X[i+2*nrows]-X[i]-X[i+4*nrows])*hxdhy-sc*exp(X[i+2*nrows]);
  }
}
