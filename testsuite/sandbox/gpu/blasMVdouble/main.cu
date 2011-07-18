#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <device_types.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include <cublas.h>


#define SEED 1
#define THREADS 128
#define BLOCKS 1024


// MATRIX_DIM_M,MATRIX_DIM_N,XTILES,YTILES
__constant__ unsigned int symbolDIMS[4];


__global__ void multMV_kernel(double*,double*,double*);
double getclock();


int main(int argc, char** argv){

        printf("\n\nStarting GPUdoubleMV with BLAS kernel...\n");
        unsigned int MATRIX_DIM_M = atoi(argv[1]);//rows
	unsigned int MATRIX_DIM_N = atoi(argv[2]);//columns
	double perturb=0.;//atof(argv[3]);


//	Get device information

	int count;
	cudaGetDeviceCount(&count);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	printf("Compute capability: %d.%d\n",prop.major,prop.minor);
	printf("Number of GPUs: %d\n",count);
	printf("Multiprocessor count: %d\n",prop.multiProcessorCount);
	printf("Clock rate: %luKhz\n",prop.clockRate/1000);
	printf("Total Global Memory: %dMB\n",prop.totalGlobalMem/1000000);
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


	srand(SEED);

	int i,j;
	double cs,ce;

	cudaError_t cudastatus0,cudastatus1,
	            cudastatus2,cudastatus3,
	            cudastatus4,cudastatus5,
           	    cudastatus6,cudastatus7;

	float elapsedtime;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudastatus0=cudaEventCreate(&start);
	printf("Status event create status0: %s\n",cudaGetErrorString(cudastatus0));
	cudaEventCreate(&stop);


//	allocate random Matrix 
	unsigned int memmatrixsize=sizeof(double)*MATRIX_DIM_M*MATRIX_DIM_N;
	double* Matrix=(double*) malloc(memmatrixsize); 
	printf("\n\nmemsize of Matrix: %fMB\n",(float)memmatrixsize/(float)1048576);

//      Dense vector
	unsigned int memvectorsize=MATRIX_DIM_N*sizeof(double);
	double* vector=(double*)malloc(memvectorsize);
	printf("memsize of vector: %luKB\n",memvectorsize/1000);
	
	unsigned int memreturnsize = MATRIX_DIM_M*sizeof(double);
	double* returnvector=(double*)malloc(MATRIX_DIM_M*sizeof(double));
	printf("memsize of return vector: %luKB\n",memreturnsize/1000);

	printf("Total data transfer to gpu: %luKB\n",(memreturnsize+memvectorsize+memmatrixsize)/1000);



//	fill Matrix and vector
//	#pragma omp parallel for private(i,j)
	for(i=0;i<MATRIX_DIM_M;i++){
           
           for(j=0;j<MATRIX_DIM_N;j++) Matrix[j*MATRIX_DIM_M+i]=(double) 1.0*(1.0/((rand()%100)+1));    
	}
	for(i=0;i<MATRIX_DIM_N;i++)
		vector[i]=(double) 1.0*(1.0/((rand()%100)+1))+perturb;

	printf("sample vector[0]: %f\n",vector[0]);
	printf("sample vector[1]: %f\n",vector[1]);
	printf("sample vector[2]: %f\n",vector[2]);
	printf("sample vector[3]: %f\n",vector[3]);
	printf("sample vector[4]: %f\n",vector[4]);
	printf("sample vector[5]: %f\n",vector[5]);
	printf("sample vector[6]: %f\n",vector[6]);

// 	allocate GPU device memory
	cs=getclock();
	double* dev_vector;
	cudastatus0=cudaMalloc((void**)&dev_vector,MATRIX_DIM_N*sizeof(double));
	cudastatus1=cudaMemcpy(dev_vector,vector,MATRIX_DIM_N*sizeof(double),cudaMemcpyHostToDevice);
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in dev_vector memory allocation:\nstatus0: %s, status1: %s\nExiting...\n\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(vector)free(vector);
	  if(Matrix)free(Matrix);
	  exit(EXIT_FAILURE);
	}


	double* dev_Matrix;
	cudastatus2=cudaMalloc((void**)&dev_Matrix,MATRIX_DIM_M*MATRIX_DIM_N*sizeof(double));
	cudastatus3=cudaMemcpy(dev_Matrix,Matrix,MATRIX_DIM_M*MATRIX_DIM_N*sizeof(double),cudaMemcpyHostToDevice);
	if(cudastatus2!=cudaSuccess|cudastatus3!=cudaSuccess){
	  printf("Error in dev_Matrix memory allocation:\nstatus2: %s, status3: %s.\nExiting...\n\n",
  			cudaGetErrorString(cudastatus2),
			cudaGetErrorString(cudastatus3));
	  if(dev_vector) cudaFree(dev_vector);
	  if(vector)free(vector);
	  if(Matrix)free(Matrix);
	  exit(EXIT_FAILURE);
	}


	double* dev_returnVector;
	cudastatus4=cudaMalloc((void**)&dev_returnVector,MATRIX_DIM_M*sizeof(double));
	cudastatus5=cudaMemset(dev_returnVector,0.0,MATRIX_DIM_M*sizeof(double));
	if(cudastatus4!=cudaSuccess|cudastatus5!=cudaSuccess){
	  printf("Error in dev_returnVector memory allocation:\nstatus4: %s, status5: %s\nExiting...\n\n",
	  		cudaGetErrorString(cudastatus4),
			cudaGetErrorString(cudastatus5));
	  if(dev_vector) cudaFree(dev_vector);
	  if(dev_Matrix) cudaFree(dev_Matrix);
	  if(vector)free(vector);
	  if(Matrix)free(Matrix);
	  exit(EXIT_FAILURE);
	}


//	update constant memory
	unsigned int ytiles=ceil((float)MATRIX_DIM_M/(float)BLOCKS);
	unsigned int xtiles=ceil((float)MATRIX_DIM_N/(float)THREADS);
	printf("ytiles: %d, xtiles: %d\n",ytiles,xtiles);
	unsigned int matrixdims[4]={MATRIX_DIM_N,MATRIX_DIM_M,xtiles,ytiles};
	cudastatus6=cudaMemcpyToSymbol("symbolDIMS",matrixdims,4*sizeof(unsigned int));
	if(cudastatus6!=cudaSuccess){
	  printf("Error in symbol copy:\nstatus6: %s.\nExiting...\n\n",
	  		cudaGetErrorString(cudastatus6));
	  if(dev_vector) cudaFree(dev_vector);
	  if(dev_Matrix) cudaFree(dev_Matrix);
	  if(dev_returnVector) cudaFree(dev_returnVector);
	  if(vector)free(vector);
	  if(Matrix)free(Matrix);
	  exit(EXIT_FAILURE);
	}
	ce=getclock();

//	set thread grid layout
	const int num_blocks=BLOCKS;
	const int num_threads_per_block=THREADS;
	printf("Set number of BLOCKS: %d, number of THREADS_PER_BLOCK: %d\n",num_blocks,num_threads_per_block);
	printf("------------------------------------------------------------\n\n");


//	start timer	
	cudaEventRecord(start,0);

//	call kernel
	multMV_kernel<<<num_blocks,num_threads_per_block>>>(dev_Matrix,dev_vector,dev_returnVector);

//	end timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop); // barrier

//	kernel time
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudastatus7=cudaMemcpy(returnvector,dev_returnVector,MATRIX_DIM_M*sizeof(double),cudaMemcpyDeviceToHost);

	if(cudastatus7!=cudaSuccess){
          printf("Error, kernel return status: %s\nExiting...\n\n",
	  		cudaGetErrorString(cudastatus7));
          if(dev_vector) cudaFree(dev_vector);
          if(dev_Matrix) cudaFree(dev_Matrix);
          if(dev_returnVector) cudaFree(dev_returnVector);
	  if(vector) free(vector);
	  if(Matrix) free(Matrix);
          exit(EXIT_FAILURE);
        }else{
	  //for(i=0;i<MATRIX_DIM_M;i++) printf("returnvector: %f\n",returnvector[i]);
	  printf("Kernel return successfully, elapsed time: %6.9lf sec.\n",elapsedtime/1000.0);
	  printf("Data set up time, elapsed time: %6.9lf sec.\n",ce-cs);
          printf("Total gpu kernel elapsed time: %6.9lf sec.\n",(elapsedtime/1000.0)+(ce-cs));
	  printf("...............................................\n\n");
	}



	printf("Calculating with openmp on CPU with 8 cores...");
	cs = getclock();
	double* wvector = (double*) calloc(MATRIX_DIM_M,sizeof(double));
	#pragma omp parallel for private(i,j)
	for(i=0;i<MATRIX_DIM_M;i++){
		for(j=0;j<MATRIX_DIM_N;j++){
			wvector[i]+=Matrix[i*MATRIX_DIM_N+j]*vector[j];
		}
	}
	ce = getclock();
	printf("finished, elapsed time: %6.9lf sec.\n",ce-cs);
	
	
	printf("Calculating on CPU in serial...");
	free(wvector);
	cs = getclock();
	wvector = (double*) calloc(MATRIX_DIM_M,sizeof(double));
	for(i=0;i<MATRIX_DIM_M;i++){
		for(j=0;j<MATRIX_DIM_N;j++){
			wvector[i]+=Matrix[i*MATRIX_DIM_N+j]*vector[j];
		}
	}
	ce = getclock();
	printf("finished, elapsed time: %6.9lf sec.\n",ce-cs);


 printf("...............................................\n\n");



// free up previous memory
	if(dev_vector) cudaFree(dev_vector);
	if(dev_Matrix) cudaFree(dev_Matrix);
	if(dev_returnVector)cudaFree(dev_returnVector);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



// CUBLAS_STATUS_NOT_INITIALIZED
// CUBLAS_STATUS_INVALID_VALUE
// CUBLAS_STATUS_MAPPING _ERROR
// CUBLAS_STATUS_SUCCESS
// CUBLAS_STATUS_EXECUTION_FAILED
// CUBLAS_STATUS_ALLOC_FAILED
// CUBLAS_STATUS_INTERNAL_ERROR





	cublasStatus cublastat;

	cs = getclock();
	cublasInit();
	cublastat= cublasAlloc(MATRIX_DIM_M*MATRIX_DIM_N,sizeof(double),(void**)&dev_Matrix);
	if(cublastat==CUBLAS_STATUS_ALLOC_FAILED)printf("Matrix allocation failed.\n");
	cublastat= cublasAlloc(MATRIX_DIM_N,sizeof(double),(void**)&dev_vector);	
	if(cublastat==CUBLAS_STATUS_ALLOC_FAILED)printf("Vector allocation failed.\n");
	cublastat= cublasAlloc(MATRIX_DIM_M,sizeof(double),(void**)&dev_returnVector);	
	if(cublastat==CUBLAS_STATUS_ALLOC_FAILED)printf("Return vector allocation failed.\n");

	cublastat = cublasSetMatrix(MATRIX_DIM_M,MATRIX_DIM_N,sizeof(double),Matrix,
		  MATRIX_DIM_M,dev_Matrix,MATRIX_DIM_M);
	if(cublastat!=CUBLAS_STATUS_SUCCESS)printf("Set Matrix failed.\n");
	cublastat = cublasSetVector(MATRIX_DIM_N,sizeof(double),vector,1,dev_vector,1);	
	if(cublastat!=CUBLAS_STATUS_SUCCESS)printf("Set vector failed.\n");
	ce = getclock();


	cudaEventRecord(start,0);

	cublasDgemv('T',MATRIX_DIM_M,MATRIX_DIM_N,1.0,dev_Matrix,MATRIX_DIM_M,
			dev_vector,1,0.0,dev_returnVector,1);


//	end timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop); // barrier
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cublastat =cublasGetError();
	if(cublastat!=CUBLAS_STATUS_SUCCESS)printf("CUBLAS kernel failed.\n");
	else{




	  printf("CUBLAS Kernel return successfully, elapsed time: %6.9lf sec.\n",elapsedtime/1000.0);
	  printf("Data set up time, elapsed time: %6.9lf sec.\n",ce-cs);
          printf("Total gpu kernel elapsed time: %6.9lf sec.\n",(elapsedtime/1000.0)+(ce-cs));
	  printf("...............................................\n");

   }







//	for(i=0;i<MATRIX_DIM_M;i++)printf("wvector[%d]: %f\n",i,wvector[i]);

	double answer=0.;
//	float cpu_ysqsum=0.;
//	float gpu_ysqsum=0.;
	unsigned int counter=0;
	printf("\n\nVerifying correct answer from gpu.\n");
	for(i=0;i<MATRIX_DIM_M;i++){
		answer=wvector[i]-returnvector[i];
	//	printf("answer: %f\n",answer);
	//	printf("CPUvector: %f\n",wvector[i]);
	//	printf("GPUvector: %f\n",returnvector[i]);
//		cpu_ysqsum+=(wvector[i]*wvector[i]);
//		gpu_ysqsum+=(returnvector[i]*returnvector[i]);
		if (answer!=0.0){	
		   printf("Error, divergent value: %f for index: %d\n",answer,i);
		   printf("CPUvector (serial): %f\n",wvector[i]);
		   printf("GPUvector: %f\n",returnvector[i]);
		   counter++;
		   if(counter>0) break;
		}
	}

//	printf("\nCPU ysqsum: %9.9lf\n",cpu_ysqsum);
//	printf("GPU ysqsum: %9.9lf\n\n",gpu_ysqsum);




	printf("Freeing memory...\n");
	if(dev_vector) cudaFree(dev_vector);
	if(dev_Matrix) cudaFree(dev_Matrix);
	if(dev_returnVector)cudaFree(dev_returnVector);
	free(vector);
	free(returnvector);
	free(wvector);
	free(Matrix);

	printf("Done, exiting...\n\n\n");
	
	exit(EXIT_SUCCESS);
}





// Y-BLOCK TILES
__global__ void multMV_kernel(double* M, double* v, double* w){

	unsigned int i,j,tileId,offset,vindex,ypos;
	__shared__ double vcache[THREADS];
	__shared__ double wback[THREADS];


	//symbolDIMS[0]== M
	//symbolDIMS[1]== N
	//symbolDIMS[2]== xtiles
	//symbolDIMS[3]== ytiles

	for(i=0;i<symbolDIMS[2];i++){

	   vindex=i*blockDim.x+threadIdx.x;

	   if(vindex<symbolDIMS[0]) vcache[threadIdx.x]=v[vindex];
	   else vcache[threadIdx.x]=0.0;
 
	   //do memory accesses
	   for(j=0;j<symbolDIMS[3];j++){
	       ypos=j*gridDim.x+blockIdx.x;
	       tileId=symbolDIMS[0]*ypos+vindex;
	       
	       if(tileId < symbolDIMS[0]*symbolDIMS[1])wback[threadIdx.x]=M[tileId]*vcache[threadIdx.x];
	       else wback[threadIdx.x]=0.;

	       __syncthreads();
	       
// 	       per block thread reduction
	       offset = blockDim.x/2;
	       while(offset>0){
	            if(threadIdx.x<offset) 
		       wback[threadIdx.x]+=wback[threadIdx.x+offset];
		    __syncthreads();
		    offset/=2;
	       }//end while
	   
//	       top level tile reduction         
	       if(threadIdx.x==0) w[ypos]+=wback[0];  
	       __syncthreads();

	    }//end Ytiles-for	
	}//end Xtiles-for
}//end function







double getclock(){

      struct timezone tzp;
      struct timeval tp;
      gettimeofday (&tp, &tzp);
      return (tp.tv_sec + tp.tv_usec*1.0e-6);

}











