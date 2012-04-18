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

  /*@ begin Loop(transform CUDA(threadCount=32, cacheBlocks=False, pinHostMem=False)
  for(i=bb; i<=be-1; i++) {
    F[i] = (2*X[i+2*nrows] - X[i+nrows] - X[i+3*nrows])*hydhx + (2*X[i+2*nrows] - X[i] - X[i+4*nrows])*hxdhy - sc*exp(X[i+2*nrows]);
  }
  ) @*/
  {
    /*declare variables*/
    double *dev_hxdhy, *dev_X, *dev_hydhx, *dev_sc, *dev_F;
    int nthreads=32;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=(be+nthreads-1)/nthreads;
    /*allocate device memory*/
    int nbytes=be*sizeof(double);
    cudaMalloc((void**)&dev_hxdhy,sizeof(double));
    cudaMalloc((void**)&dev_sc,sizeof(double));
    cudaMalloc((void**)&dev_hydhx,sizeof(double));
    cudaMalloc((void**)&dev_X,nbytes);
    cudaMalloc((void**)&dev_F,nbytes);
    /*copy data from host to device*/
    cudaMemcpy(dev_hxdhy,&hxdhy,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sc,&sc,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_hydhx,&hydhx,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_X,X,nbytes,cudaMemcpyHostToDevice);
    /*invoke device kernel*/
    orio_t_start=getClock();
    orcu_kernel3<<<dimGrid,dimBlock>>>(be,nrows,bb,dev_hxdhy,dev_X,dev_hydhx,dev_sc,dev_F);
    /*copy data from device to host*/
    cudaMemcpy(F,dev_F,nbytes,cudaMemcpyDeviceToHost);
    /*free allocated memory*/
    cudaFree(dev_hxdhy);
    cudaFree(dev_X);
    cudaFree(dev_hydhx);
    cudaFree(dev_sc);
    cudaFree(dev_F);
  }
/*@ end @*/
}
__global__ void orcu_kernel3(int be, int nrows, int bb, double* hxdhy, double* X, double* hydhx, double* sc, double* F) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x+bb;
  if (tid<=be-1) {
    F[tid]=(2*X[tid+2*nrows]-X[tid+nrows]-X[tid+3*nrows])*(*hydhx)+(2*X[tid+2*nrows]-X[tid]-X[tid+4*nrows])*(*hxdhy)-(*sc)*exp(X[tid+2*nrows]);
  }
}
