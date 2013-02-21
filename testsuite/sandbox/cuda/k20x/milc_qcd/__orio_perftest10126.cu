/*
CFLAGS:-O3
UIF:5
TC:448
PL:48
BC:70
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

#define SITES 2
double *A;
double *x;
double *y;
void malloc_arrays() {
  int i1;
  A = (double*) malloc((18 *SITES) * sizeof(double));
  x = (double*) malloc((6 *SITES) * sizeof(double));
  y = (double*) malloc((6 *SITES) * sizeof(double));
}

void init_input_vars() {
  int i1;
  for (i1=0; i1<18 *SITES; i1++)
   A[i1] = (i1) % 5 + 1;
  for (i1=0; i1<6 *SITES; i1++)
   x[i1] = (i1) % 5 + 1;
  for (i1=0; i1<6 *SITES; i1++)
   y[i1] = 0;
}



__global__ void orcu_kernel30377(const int sites_on_node, double* A, double* y, double* x) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int gsize=gridDim.x*blockDim.x;
  double ci, ai, bi, ar, br, cr;
  int j, k;
  for (int i=tid; i<=sites_on_node-1; i+=gsize) {
    {
      #pragma unroll 5
      for (j=0; j<=5; j=j+2) {
        cr=ci=0.0;
        for (k=0; k<=5; k=k+2) {
          ar=A[18*i+3*j+k];
          ai=A[18*i+3*j+k+1];
          br=x[6*i+k];
          bi=x[6*i+k+1];
          cr=cr+ar*br-ai*bi;
          ci=ci+ar*bi+ai*br;
        }
        y[6*i+j]=cr;
        y[6*i+j+1]=ci;
      }
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
    

  int sites_on_node=SITES;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL, unrollInner=UIF)

  for(i=0; i<=sites_on_node-1; i++) {
    for(j=0; j<=5; j+=2) {
      cr = ci = 0.0;
      for(k=0; k<=5; k+=2) {
        ar=A[18*i+3*j+k];
        ai=A[18*i+3*j+k+1];
        br=x[6*i+k];
        bi=x[6*i+k+1];
        cr += ar*br - ai*bi;
        ci += ar*bi + ai*br;
      }
      y[6*i+j]  =cr;
      y[6*i+j+1]=ci;
    }
  }

  ) @*/
  {
    cudaDeviceSynchronize();
    /*declare variables*/
    double *dev_A, *dev_y, *dev_x;
    int nthreads=448;
    /*calculate device dimensions*/
    dim3 dimGrid, dimBlock;
    dimBlock.x=nthreads;
    dimGrid.x=70;
    /*allocate device memory*/
    cudaMalloc(&dev_A,18 *SITES*sizeof(double));
    cudaMalloc(&dev_x,6 *SITES*sizeof(double));
    cudaMalloc(&dev_y,6 *SITES*sizeof(double));
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    /*copy data from host to device*/
    cudaEventRecord(tstart,0);
    cudaMemcpy(dev_A,A,18 *SITES*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x,x,6 *SITES*sizeof(double),cudaMemcpyHostToDevice);
    cudaEventRecord(tstop,0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&orcu_transfer,tstart,tstop);
    cudaEventRecord(start,0);
    /*invoke device kernel*/
    orcu_kernel30377<<<dimGrid,dimBlock>>>(sites_on_node,dev_A,dev_y,dev_x);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&orcu_elapsed,start,stop);
    /*copy data from device to host*/
    cudaMemcpy(y,dev_y,6 *SITES*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    /*free allocated memory*/
    cudaFree(dev_A);
    cudaFree(dev_y);
    cudaFree(dev_x);
    cudaError_t err=cudaGetLastError();
    if (cudaSuccess!=err) 
      printf("CUDA runtime error: %s@",cudaGetErrorString(err));
  }
/*@ end @*/
  
    printf("{'[13, 4, 4, 1, 3]' : (%g,%g)}\n", orcu_elapsed, orcu_transfer);
  }
  cudaEventDestroy(tstart); cudaEventDestroy(tstop);
  cudaEventDestroy(start);  cudaEventDestroy(stop);
  
  
  return 0;
}
