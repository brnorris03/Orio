//scalarvectorN.cu
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
//#include <omp.h>


#define SEED 1

double getclock();
void checkCUDAError(const char *);
__global__ void sVPN(float *vector, float* scalar, int* vlength, int *N);
__global__ void sVPNo(float *vector, float* scalar, int* vlength, int *N);






int main(int argc, char** argv){

        printf("\n\nARGC value: %d\n",argc);
        int N = atoi(argv[1]);
	int len = atoi(argv[2]);
	int tpb = atoi(argv[3]);
	if(argc<5){
	  srand(SEED);
	}else{
	  srand(atoi(argv[4]));
	}

	
	int i;


//	Get device information
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

//	some general cpu info
	printf("size of float (cpu): %d\n",sizeof(float));
	printf("size of unsigned int (cpu): %d\n",sizeof(unsigned int));
	printf("size of unsigned long (cpu): %d\n",sizeof(unsigned long));


	float scalar=100.00/(rand()%100+1);

	float *vector;
	vector=(float*)malloc(len*sizeof(float));

	for(i=0;i<len;i++){
	    vector[i]=100.00/(rand()%100+1);
	}

	cudaError_t cudastatus0,cudastatus1;

	float* devVec;
	cudastatus0=cudaMalloc((void**)&devVec,len*sizeof(float));
	cudastatus1=cudaMemcpy(devVec,vector,len*sizeof(float),cudaMemcpyHostToDevice);
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in devVec memory allocation:\nstatus0: %s, status1: %s\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(devVec) cudaFree(devVec);
	  if(vector) free(vector);
          exit(1);
	}

	float* devScal;
	cudastatus0=cudaMalloc((void**)&devScal,sizeof(float));
	cudastatus1=cudaMemcpy(devScal,&scalar,sizeof(float),cudaMemcpyHostToDevice);
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in devScal memory allocation:\nstatus0: %s, status1: %s\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(devVec) cudaFree(devVec);
	  if(devScal) cudaFree(devScal);
	  if(vector) free(vector);
          exit(1);
	}

	int* devLen;
	cudastatus0=cudaMalloc((void**)&devLen,sizeof(int));
	cudastatus1=cudaMemcpy(devLen,&len,sizeof(int),cudaMemcpyHostToDevice);
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in devLen memory allocation:\nstatus0: %s, status1: %s\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(devVec) cudaFree(devVec);
	  if(devScal) cudaFree(devScal);
	  if(vector) free(vector);
          exit(1);
	}

	int* devN;
	cudastatus0=cudaMalloc((void**)&devN,sizeof(int));
	cudastatus1=cudaMemcpy(devN,&N,sizeof(int),cudaMemcpyHostToDevice);
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in devN memory allocation:\nstatus0: %s, status1: %s\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(devVec) cudaFree(devVec);
	  if(devScal) cudaFree(devScal);
	  if(devLen) cudaFree(devLen);
	  if(devN) cudaFree(devN);
	  if(vector) free(vector);
          exit(1);
	}


	printf("\n\nVector size: %dK Iterations: %dK Memory: %dKB\n",len/1000,N/1000,len*sizeof(float)/1024);

	int blocks=ceil((float)len/(float)tpb);
	int threads=tpb;
	printf("Number of blocks: %d, threads per block: %d\n",blocks,threads);

	cudastatus0=cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	if(cudastatus0!=cudaSuccess){
	  printf("Error in setting L1 cache level to prefer Shared, status0: %s\n",
	  		cudaGetErrorString(cudastatus0));
	}

	float elapsedtime1;
	cudaEvent_t start1,stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1,0);//begin recording kernel
  	sVPN<<<blocks,threads>>>(devVec,devScal,devLen,devN);
	cudaEventRecord(stop1,0);
	cudaEventSynchronize(stop1); // event barrier
	cudaEventElapsedTime(&elapsedtime1,start1,stop1);
        cudaEventDestroy(start1);
	cudaEventDestroy(stop1);


	float elapsedtime2;
	cudaEvent_t start2,stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2,0);//begin recording kernel
  	sVPNo<<<blocks,threads>>>(devVec,devScal,devLen,devN);
	cudaEventRecord(stop2,0);
	cudaEventSynchronize(stop2); // event barrier
	cudaEventElapsedTime(&elapsedtime2,start2,stop2);
        cudaEventDestroy(start2);
	cudaEventDestroy(stop2);


	cudastatus0=cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	if(cudastatus0!=cudaSuccess){
	  printf("Error in setting L1 cache level to Cache preferred, status0: %s\n",
	  		cudaGetErrorString(cudastatus0));
	}

	float elapsedtime3;
	cudaEvent_t start3,stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventRecord(start3,0);//begin recording kernel
  	sVPN<<<blocks,threads>>>(devVec,devScal,devLen,devN);
	cudaEventRecord(stop3,0);
	cudaEventSynchronize(stop3); // event barrier
	cudaEventElapsedTime(&elapsedtime3,start3,stop3);
        cudaEventDestroy(start3);
	cudaEventDestroy(stop3);


	float elapsedtime4;
	cudaEvent_t start4,stop4;
	cudaEventCreate(&start4);
	cudaEventCreate(&stop4);
	cudaEventRecord(start4,0);//begin recording kernel
  	sVPNo<<<blocks,threads>>>(devVec,devScal,devLen,devN);
	cudaEventRecord(stop4,0);
	cudaEventSynchronize(stop4); // event barrier
	cudaEventElapsedTime(&elapsedtime4,start4,stop4);
        cudaEventDestroy(start4);
	cudaEventDestroy(stop4);



	printf("\nunoptimized kernel 16KB L1: %lf msec.\n",elapsedtime1);
	printf("optimized kernel 16KB L1: %lf msec.\n",elapsedtime2);
	printf("unoptimized kernel 48KB L1: %lf msec.\n",elapsedtime3);
	printf("optimized kernel 48KB L1: %lf msec.\n\n\n",elapsedtime4);

	if(devScal) cudaFree(devScal);
	if(devLen) cudaFree(devLen);
	if(devN) cudaFree(devN);
	if(devVec) cudaFree(devVec);
	if(vector) free(vector);
	checkCUDAError("cuda free operations");
	printf("Exiting...\n\n");
	exit(0);
}




// .................................................................
// Unoptimized code - kernel multiplies a vector by a scalar N times
// .................................................................
//
__global__ void sVPN(float *vector, float* scalar, int* vlength, int *N){

  int tid = blockDim.x*blockIdx.x+threadIdx.x;

  unsigned int i;// using unsigned type for loop counter will decrease performance
  
  for(i=0;i<*N;i++){// per loop global access of N if not in register
  	if(tid<*vlength)vector[tid]*=*scalar;//per loop global access of scalar and vlength 
  }
  
  //3N global memory accesses per thread or 4N if N is not in register
}




// .................................................................
// Optimized code - kernel multiplies a vector by a scalar N times
// 	 replacing global memory access with register(local) variables
// .................................................................
//
__global__ void sVPNo(float *vector, float* scalar, int* vlength, int *N){

  int tid = blockDim.x*blockIdx.x+threadIdx.x;

  //local variables added
  float local_scalar = *scalar;
  float local_N = *N;
  int local_vlength=*vlength;
  int local_vector= vector[tid];
  
  int i;// transform to int
  
  for(i=0;i< local_N;i++){
  	if(tid<local_vlength)local_vector*=local_scalar; 
  }
  vector[tid]=local_vector;
  //5 global memory accesses per thread
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
