
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

#define M 64
#define N 64
#define P 64
#define NOS 7
#define DOF 1
double *A;
double *x;
double *y;
int offsets[NOS]={-M *N *DOF ,-M *DOF ,-DOF ,0 ,DOF ,M *DOF ,M *N *DOF };
void malloc_arrays() {
  int i1;
  A = (double*) malloc((M *N *P *DOF *DOF *NOS) * sizeof(double));
  x = (double*) malloc((M *N *P *DOF) * sizeof(double));
  y = (double*) malloc((M *N *P *DOF) * sizeof(double));
}

void init_input_vars() {
  int i1;
  for (i1=0; i1<M *N *P *DOF *DOF *NOS; i1++)
   A[i1] = (i1) % 5 + 1;
  for (i1=0; i1<M *N *P *DOF; i1++)
   x[i1] = (i1) % 5 + 1;
  for (i1=0; i1<M *N *P *DOF; i1++)
   y[i1] = 0;
}



__global__ void orcu_kernel2(const int nrows, const int ndiags, int sbdiag, int ndofs, int* offsets, double* A, double* x, double* y) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  double ysum;
  int j, k, col, row;
  for (int i=tid; i<=nrows-1; i+=gsize) {
    {
      ysum=0.0;
      for (j=0; j<=ndiags-1; j++ ) {
        row=i+j*sbdiag;
        col=(floor((float)i/ndofs)+offsets[j])*ndofs;
        if (col>=0&&col<nrows) 
          for (k=0; k<=ndofs-1; k++ ) 
            ysum=ysum+A[row+k*nrows]*x[col+k];
      }
      y[i]=ysum;
    }
  }
}


int main(int argc, char *argv[]) {
  malloc_arrays();
  init_input_vars();

  cudaSetDeviceFlags(cudaDeviceBlockingSync);
  float orcu_elapsed=0.0, orcu_transfer=0.0;
  cudaEvent_t tstart, tstop, start, stop;
  cudaEventCreate(&tstart); cudaEventCreate(&tstop);
  cudaEventCreate(&start);  cudaEventCreate(&stop);
  for (int orio_i=0; orio_i<ORIO_REPS; orio_i++) {
    

  int nrows=M*N*P*DOF;
  int ndiags=NOS;
  int ndofs=DOF;
  int sbdiag=M*N*P*DOF*DOF;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL)

  for(i=0; i<=nrows-1; i++){
    ysum = 0.0;
    for(j=0; j<=ndiags-1; j++){
      row = i+j*sbdiag;
      col = (floor((float)i/ndofs)+offsets[j])*ndofs;
      if(col>=0&&col<nrows)
        for(k=0; k<=ndofs-1; k++)
          ysum += A[row+k*nrows] * x[col+k];
    }
    y[i] = ysum;
  }

  ) @*/
  {
    cudaDeviceSynchronize();
    /*declare variables*/
    double *dev_A, *dev_x, *dev_y;
    int *dev_offsets;
    int nthreads=32;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=14;
    /*allocate device memory*/
    cudaMalloc(&dev_A,M *N *P *DOF *DOF *NOS*sizeof(double));
    cudaMalloc(&dev_x,M *N *P *DOF*sizeof(double));
    cudaMalloc(&dev_y,M *N *P *DOF*sizeof(double));
    cudaMalloc(&dev_offsets,NOS*sizeof(int));
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    /*copy data from host to device*/
    cudaEventRecord(tstart,0);
    cudaMemcpy(dev_A,A,M *N *P *DOF *DOF *NOS*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x,x,M *N *P *DOF*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_offsets,offsets,NOS*sizeof(int),cudaMemcpyHostToDevice);
    cudaEventRecord(tstop,0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&orcu_transfer,tstart,tstop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    orcu_kernel2<<<dimGrid,dimBlock>>>(nrows,ndiags,sbdiag,ndofs,dev_offsets,dev_A,dev_x,dev_y);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    /*copy data from device to host*/
    cudaMemcpy(y,dev_y,M *N *P *DOF*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    /*free allocated memory*/
    cudaFree(dev_A);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_offsets);
    cudaError_t err=cudaGetLastError();
    if (cudaSuccess!=err) 
      printf("CUDA runtime error: %s@",cudaGetErrorString(err));
  }
/*@ end @*/
  
    printf("{'[0, 0, 0]' : (%g,%g)}\n", orcu_elapsed, orcu_transfer);
  }
  cudaEventDestroy(tstart); cudaEventDestroy(tstop);
  cudaEventDestroy(start);  cudaEventDestroy(stop);
  
  
  return 0;
}
