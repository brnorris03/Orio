//dot.cu
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>


#define SEED 1
#define SHARESIZE 128

double getclock();
void checkCUDAError(const char *);
__global__ void dot(float* x, float* y, float* z, int* vlength);
__global__ void doto(float* x, float* y, float* z, int* vlength);




int main(int argc, char** argv){

        printf("\n\nARGC value: %d\n",argc);
        int len = atoi(argv[1]);
	if(argc<3){
	  srand(SEED);
	}else{
	  srand(atoi(argv[2]));
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


	float *X;
	X=(float*)malloc(len*sizeof(float));

	float* Y;
	Y=(float*)malloc(len*sizeof(float));

	for(i=0;i<len;i++){
	    X[i]=100.00/(rand()%100+1);
	    Y[i]=100.00/(rand()%100+1);
	}

	cudaError_t cudastatus0,cudastatus1;

	float* devX;
	cudastatus0=cudaMalloc((void**)&devX,len*sizeof(float));
	cudastatus1=cudaMemcpy(devX,X,len*sizeof(float),cudaMemcpyHostToDevice);
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in devX memory allocation:\nstatus0: %s, status1: %s\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(devX) cudaFree(devX);
	  if(X) free(X);
	  if(Y) free(Y);
          exit(1);
	}

	float* devY;
	cudastatus0=cudaMalloc((void**)&devY,len*sizeof(float));
	cudastatus1=cudaMemcpy(devY,Y,len*sizeof(float),cudaMemcpyHostToDevice);
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in devY memory allocation:\nstatus0: %s, status1: %s\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(devX) cudaFree(devX);
	  if(devY) cudaFree(devY);
	  if(X) free(X);
	  if(Y) free(Y);
          exit(1);
	}

	float* devZ;
	cudastatus0=cudaMalloc((void**)&devZ,sizeof(float));
	cudastatus1=cudaMemset(devZ,0.0,sizeof(float));
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in devZ memory allocation:\nstatus0: %s, status1: %s\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(devX) cudaFree(devX);
	  if(devY) cudaFree(devY);
	  if(devZ) cudaFree(devZ);
	  if(X) free(X);
	  if(Y) free(Y);
          exit(1);
	}

	int* devLen;
	cudastatus0=cudaMalloc((void**)&devLen,sizeof(int));
	cudastatus1=cudaMemcpy(devLen,&len,sizeof(int),cudaMemcpyHostToDevice);
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in devLen memory allocation:\nstatus0: %s, status1: %s\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(devX) cudaFree(devX);
	  if(devY) cudaFree(devY);
	  if(devZ) cudaFree(devZ);
	  if(devLen) cudaFree(devLen);
	  if(X) free(X);
	  if(Y) free(Y);
          exit(1);
	}


	printf("\n\nVector sizes: %dK, Memory: %dKB\n",len/1000,2*len*sizeof(float)/1024);

	int blocks=ceil((float)len/(float)SHARESIZE);
	int threads=SHARESIZE;
	printf("Number of blocks: %d, threads per block: %d\n",blocks,threads);


	if(prop.major>1){
		cudastatus0=cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	        if(cudastatus0!=cudaSuccess){
        	printf("Error in setting L1 cache level to prefer Shared, status0: %s\n",
                        cudaGetErrorString(cudastatus0));
        	}
	}


	float elapsedtime1;
	cudaEvent_t start1,stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1,0);//begin recording kernel
  	dot<<<blocks,threads>>>(devX,devY,devZ,devLen);
	cudaEventRecord(stop1,0);
	cudaEventSynchronize(stop1); // event barrier
	cudaEventElapsedTime(&elapsedtime1,start1,stop1);
        cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	checkCUDAError("kernel 1");
	if(prop.major>1)printf("\nunoptimized kernel 16K L1: %lf msec.\n",elapsedtime1);
	else printf("\nunoptimized kernel: %lf msec.\n",elapsedtime1);


	cudastatus0=cudaMemset(devZ,0.0,sizeof(float));
	if(cudastatus0!=cudaSuccess) printf("Error in devZ memory set, status: %s\n",
  			cudaGetErrorString(cudastatus0));
	dim3 dimGrid(1,blocks);
	dim3 dimBlock(1,threads);
	float elapsedtime2;
	cudaEvent_t start2,stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2,0);//begin recording kernel
  	doto<<<dimGrid,dimBlock>>>(devX,devY,devZ,devLen);
	cudaEventRecord(stop2,0);
	cudaEventSynchronize(stop2); // event barrier
	cudaEventElapsedTime(&elapsedtime2,start2,stop2);
        cudaEventDestroy(start2);
	cudaEventDestroy(stop2);
	checkCUDAError("kernel 2");
	if(prop.major>1)printf("optimized kernel 16K L1: %lf msec.\n",elapsedtime2);
        else printf("optimized kernel: %lf msec.\n",elapsedtime2);



	if(prop.major>1){
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
        	dot<<<blocks,threads>>>(devX,devY,devZ,devLen);
	        cudaEventRecord(stop3,0);
        	cudaEventSynchronize(stop3); // event barrier
	        cudaEventElapsedTime(&elapsedtime3,start3,stop3);
        	cudaEventDestroy(start3);
	        cudaEventDestroy(stop3);
        	checkCUDAError("kernel 3");
	        printf("\nunoptimized kernel 48K L1: %lf msec.\n",elapsedtime3);


        	cudastatus0=cudaMemset(devZ,0.0,sizeof(float));
	        if(cudastatus0!=cudaSuccess) printf("Error in devZ memory set, status: %s\n",
                        cudaGetErrorString(cudastatus0));
        	float elapsedtime4;
	        cudaEvent_t start4,stop4;
        	cudaEventCreate(&start4);
	        cudaEventCreate(&stop4);
        	cudaEventRecord(start4,0);//begin recording kernel
	        doto<<<dimGrid,dimBlock>>>(devX,devY,devZ,devLen);
        	cudaEventRecord(stop4,0);
	        cudaEventSynchronize(stop4); // event barrier
        	cudaEventElapsedTime(&elapsedtime4,start4,stop4);
	        cudaEventDestroy(start4);
	        cudaEventDestroy(stop4);
	        checkCUDAError("kernel 4");
        	printf("optimized kernel 48K L1: %lf msec.\n\n\n",elapsedtime4);
	}




	if(devX) cudaFree(devX);
	if(devY) cudaFree(devY);
	if(devZ) cudaFree(devZ);
	if(devLen) cudaFree(devLen);
	if(X) free(X);
	if(Y) free(Y);
	printf("Exiting...\n\n");
	exit(0);
}



//................................................................
// Unoptimized code - dot product using global memory
// 
//................................................................
//
__global__ void dot(float* x, float* y, float* z, int* vlength){

	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	int offset=(*vlength+1)/2;// 1 global read per thread
	
	if(tid<*vlength) y[tid]=x[tid]*y[tid];//4 global accesses per thread
	__syncthreads();
	while(offset>1){//reduction phase

		if(tid<offset) y[tid]+=y[tid+offset];//(3*vlength/2) global accesses (upper bound), 0 (lower bound)
		if(tid==offset)y[tid]=0.; //1 global access
		__syncthreads();
		
		offset=(offset+1)/2;
	}
	
	if(tid==0)*z=y[0]+y[1];// 3 global accesses
	
	// (19*vlength/2) global accesses per thread (upper)
}






//................................................................
// Optimized code - dot product using global memory
//................................................................
//
__global__ void doto(float* x, float* y, float* z, int* vlength){

	int local_vlen=*vlength;// replacment auto-variable

	int tid = blockDim.x*blockIdx.x+threadIdx.x;//becomes redundant
	
	int offset=SHARESIZE/2;// modified var

	// new code section
	int i=0;
	__shared__ float ys[SHARESIZE];//fixed size defined with macro
	int blockthreadId=threadIdx.x;//index local to threads in a block
	int gridthreadId = blockDim.x*blockIdx.x+threadIdx.x;// this is redundant but perhaps needed

	//if read is aligned and size half warp (or warp for Fermi) read is simultaneous (coalesced)
	if(gridthreadId<local_vlen) ys[blockthreadId]=y[gridthreadId];//1 global access per thread
	else ys[blockthreadId]=0.;
		
	if(gridthreadId<local_vlen) ys[blockthreadId]=x[gridthreadId]*ys[blockthreadId];//1 global accesses per thread

	__syncthreads();
	while(offset>0){//reduction phase
		if(blockthreadId<offset) ys[blockthreadId]+=ys[blockthreadId+offset];// 0 global
		__syncthreads();
		offset=offset/2;
	}
	
	if(blockthreadId==0){
		
		/*below assuming Compute capability < 2.0*/
		while(i<gridDim.x){//sequential section
		      if(i==blockIdx.x) *z+=ys[0];
		      i++;
		}
	}
	// 4 global accesses per thread (upper bound)
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
