#include "book.h"

#define UF 4

__global__ void orcu_kernel6(int n, int orcu_var3, double a1, double a2, double* y, double* x1, double* x2) {
  int tid=UF*(blockIdx.x*blockDim.x+threadIdx.x)+orcu_var3;
  if (tid<=n-UF) {
    {
      y[tid]=y[tid]+a1*x1[tid]+a2*x2[tid];
      int index = tid+1;
      y[index]=y[index]+a1*x1[index]+a2*x2[index];
      index = tid+2;
      y[index]=y[index]+a1*x1[index]+a2*x2[index];
      index = tid+3;
      y[index]=y[index]+a1*x1[index]+a2*x2[index];
    }
  }
}
//__global__ void orcu_kernel11(int n, int orcu_var8, double a1, double* y, double* x1) {
  //int tid=blockIdx.x*blockDim.x+threadIdx.x+orcu_var8;
  //if (tid<=n-1) {
    //y[tid]=y[tid]+a1*x1[tid];
  //}
//}


void axpy1(int n, double *y, double a1, double a2, double *x1, double *x2)
{
register int i;


/*@ begin Loop(
  transform Composite(
    cuda = (16,False, False, 1)
    ,scalarreplace = (False, 'int')
, unrolljam = (['i'], [2])
  )
   {
    for (i=0; i<=n-1; i++) {
    	y[i]=y[i]+a1*x1[i];
    }
    
   }


   
  
) @*/

cudaEvent_t start, stop;
HANDLE_ERROR(cudaEventCreate(&start));
HANDLE_ERROR(cudaEventCreate(&stop));


{
  {
    int orio_lbound1=0;
    //{
      /*declare variables*/
      double *dev_y, *dev_x1, *dev_x2;
      int nthreads=TC;
      /*calculate device dimensions*/
      dim3 dimGrid, dimBlock;
      dimBlock.x=nthreads;
      dimGrid.x=(n+nthreads-1)/nthreads;
      dimGrid.x=(dimGrid.x+UF-1)/UF;
      printf("num of blocks: %d\n", dimGrid.x);
      /*allocate device memory*/
      int nbytes=n*sizeof(double);
      cudaMalloc((void**)&dev_y,nbytes);
      cudaMalloc((void**)&dev_x1,nbytes);
      cudaMalloc((void**)&dev_x2,nbytes);
      /*copy data from host to device*/
      cudaMemcpy(dev_y,y,nbytes,cudaMemcpyHostToDevice);
      cudaMemcpy(dev_x1,x1,nbytes,cudaMemcpyHostToDevice);
      cudaMemcpy(dev_x2,x2,nbytes,cudaMemcpyHostToDevice);
      /*invoke device kernel*/
      int orcu_var3=orio_lbound1;

      HANDLE_ERROR(cudaEventRecord(start, 0));
      orcu_kernel6<<<dimGrid,dimBlock>>>(n,orcu_var3,a1,a2,dev_y,dev_x1, dev_x2);
      HANDLE_ERROR(cudaEventRecord(stop, 0));
      /*copy data from device to host*/
      cudaMemcpy(y,dev_y,nbytes,cudaMemcpyDeviceToHost);
      /*free allocated memory*/
      cudaFree(dev_y);
      cudaFree(dev_x1);
      cudaFree(dev_x2);
    //}
    //int orio_lbound2=n-((n-(0))%2);
    {
      /*declare variables*/
      //double *dev_y, *dev_x1;
      //int nthreads=TC;
      /*calculate device dimensions*/
      //dim3 dimGrid, dimBlock;
      //dimBlock.x=nthreads;
      //dimGrid.x=(n+nthreads-1)/nthreads;
      /*allocate device memory*/
      //int nbytes=n*sizeof(double);
      //cudaMalloc((void**)&dev_y,nbytes);
      //cudaMalloc((void**)&dev_x1,nbytes);
      /*copy data from host to device*/
      //cudaMemcpy(dev_y,y,nbytes,cudaMemcpyHostToDevice);
      //cudaMemcpy(dev_x1,x1,nbytes,cudaMemcpyHostToDevice);
      /*invoke device kernel*/
      //int orcu_var8=orio_lbound2;
      //orcu_kernel11<<<dimGrid,dimBlock>>>(n,orcu_var8,a1,dev_y,dev_x1);
      /*copy data from device to host*/
      //cudaMemcpy(y,dev_y,nbytes,cudaMemcpyDeviceToHost);
      /*free allocated memory*/
      //cudaFree(dev_y);
      //cudaFree(dev_x1);
    }
  }
}
/*@ end @*/
HANDLE_ERROR(cudaEventSynchronize(stop));
float passedTime;
HANDLE_ERROR(cudaEventElapsedTime(&passedTime, start, stop));
HANDLE_ERROR(cudaEventDestroy(start));
HANDLE_ERROR(cudaEventDestroy(stop));
printf("timePassed: %f ms\n", passedTime);
}

int main(){
	double* y = (double*) malloc(sizeof(double)*NN);
	double* x1 = (double*) malloc(sizeof(double)*NN);
	double* x2 = (double*) malloc(sizeof(double)*NN);
	double a1 = AA;
	double a2 = AA2;
	int i;
        
	for(i=0; i<NN; i++){
		y[i] = i;
		x1[i] = i;
		x2[i] = i;
	}
	axpy1(NN, y, a1, a2, x1, x2);
	for(i=0; i<13; i++)
		printf("%f\n", y[i]);
        for(i=NN-9; i<NN; i++)
                printf("%f\n", y[i]);

	return 0;
}
