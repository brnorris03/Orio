#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <stdarg.h>
//#include <omp.h>


#define SEED 1
#define VERBOSE 0

double getclock();
void checkCUDAError(const char *);
extern void freeAllmem();
extern void init_kernel(int numarray,...);


int main(int argc, char** argv){

	
	if(VERBOSE){//Get device information
	      int count;
	      cudaGetDeviceCount(&count);
	      cudaDeviceProp prop;
	      cudaGetDeviceProperties(&prop,0);
	      printf("Compute capability: %d.%d\n",prop.major,prop.minor);
	      printf("Number of GPUs: %d\n",count);
	      printf("Multiprocessor count: %d\n",prop.multiProcessorCount);
	      printf("Clock rate: %luKhz\n",prop.clockRate/1000);
	      printf("Total Global Memory: %luMB\n",(unsigned int)prop.totalGlobalMem/1000000);
	      printf("Total Constant Memory: %d\n",prop.totalConstMem);
	      printf("Shared memory per block: %d\n",prop.sharedMemPerBlock);
	      printf("1-D Texture Max size: %d\n",prop.maxTexture1D);
	      printf("Number of registers per block: %d\n",prop.regsPerBlock);
	      printf("Can I map host memory: %d\n",prop.canMapHostMemory);
	      printf("Max number of threads per block: %d\n",prop.maxThreadsPerBlock);
	      printf("Max number of blocks in a grid [0]: %d\n",prop.maxGridSize[0]);
	      printf("Max number of blocks in a grid [1]: %d\n",prop.maxGridSize[1]);
	      printf("Max number of blocks in a grid [2]: %d\n",prop.maxGridSize[2]);
	      printf("Max Texture dimensions 2D: %lu\n",prop.maxTexture2D[2]);
	      printf("Concurrent Kernels: %d\n",prop.concurrentKernels);
	      printf("Threads in a warp: %d\n",prop.warpSize);

	      //some general cpu info
	      printf("size of float (cpu): %d\n",sizeof(float));
	      printf("size of double (cpu): %d\n",sizeof(double));
	      printf("size of unsigned int (cpu): %d\n",sizeof(unsigned int));
	      printf("size of unsigned long (cpu): %d\n",sizeof(unsigned long));
	      printf("..................................\n\n");
	}


	int i;
	double cstart,cend,celapsed;
	float elapsedtime;        // CUDA device timer
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
        cudaEventCreate(&stop);


	dim3 blocks;
	dim3 threads;

//	allocate and transfer data
	init_kernel(8,0,2,2,300,0,400,1,1000);
	printf("Successfully allocated device memory.\n\n");
	


//for-loop in number of configs.
//	blocks.x=?;
//	blocks.y=?;
//	blocks.z=?;
//	threads.x=?;
//	threads.y=?;
//	threads.z=?;
//	cstart=getclock();
//	kernel_name<<<blocks,threads,1>>>(<kernel_args>);
//	cudaEventRecord(stop,0);
//	cudaEventSynchronize(stop); // event barrier
//	cend=getclock();
//	celapsed=cend-cstart;
//	cudaEventElapsedTime(&elapsedtime,start,stop);
//end for-loop

        cudaEventDestroy(start);
	cudaEventDestroy(stop);

	freeAllmem();
	exit(EXIT_SUCCESS);
}






void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 



double getclock(){
      struct timezone tzp;
      struct timeval tp;
      gettimeofday (&tp, &tzp);
      return (tp.tv_sec + tp.tv_usec*1.0e-6);
}




