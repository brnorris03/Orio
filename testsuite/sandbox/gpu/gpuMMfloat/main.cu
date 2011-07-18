#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_types.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>


#define SEED 1

#define TILE_WIDTH 16 




// MATRIX_DIM_M,MATRIX_DIM_N,XTILES,YTILES
__constant__ unsigned int symbolDIMS[4];


__global__ void multMM_kernel(float*,float*,float*);
double getclock();


int main(int argc, char** argv){

        printf("\n\nStarting GPUfloatMV...\n");
        unsigned int MATRIX_DIM_M = atoi(argv[1]);
	unsigned int MATRIX_DIM_N = atoi(argv[2]);
	unsigned int MATRIX_DIM_O = atoi(argv[3]);

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

	srand(SEED);

	int i,j,k;
	double cs,ce;

	cudaError_t cudastatus0,cudastatus1,
	            cudastatus2,cudastatus3,
	            cudastatus4,cudastatus5,
           	    cudastatus6,cudastatus7;

	float elapsedtime;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
//	allocate random MatrixA 
	unsigned int memsizeA=MATRIX_DIM_M*MATRIX_DIM_N;
	float* MatrixA=(float*) malloc(memsizeA*sizeof(float)); 
	printf("\n\nmemsize of MatrixA: %luMB\n",(sizeof(float)*MATRIX_DIM_M*MATRIX_DIM_N)/1000000);

//	allocate random MatrixB 
	unsigned int memsizeB=MATRIX_DIM_N*MATRIX_DIM_O;
	float* MatrixB=(float*) malloc(memsizeB*sizeof(float)); 
	printf("memsize of MatrixB: %luMB\n",(sizeof(float)*memsizeB)/1000000);

//	allocate empty returnMatrix 
	unsigned int memsizeC=MATRIX_DIM_M*MATRIX_DIM_O;
	float* returnMatrix=(float*) malloc(memsizeC*sizeof(float)); 
	printf("memsize of returnMatrix: %luMB\n",(sizeof(float)*memsizeC)/1000000);

	printf("Total memsize of matrices: %luMB\n\n",(sizeof(float)*(memsizeA+memsizeB+memsizeC))/1000000);


//	fill Matrices
	for(i=0;i<MATRIX_DIM_M;i++){
           for(j=0;j<MATRIX_DIM_N;j++)
	       MatrixA[j*MATRIX_DIM_M+i]=(float) 1.0*(1.0/((rand()%100)+1));    	       
	}

	for(i=0;i<MATRIX_DIM_N;i++){
           for(j=0;j<MATRIX_DIM_O;j++)
	       MatrixB[j*MATRIX_DIM_N+i]=(float) 1.0*(1.0/((rand()%100)+1));    	       
	}


// 	allocate GPU device memory
	cs=getclock();
	float* dev_MatrixA;
	cudastatus0=cudaMalloc((void**)&dev_MatrixA,MATRIX_DIM_M*MATRIX_DIM_N*sizeof(float));
	cudastatus1=cudaMemcpy(dev_MatrixA,MatrixA,MATRIX_DIM_M*MATRIX_DIM_N*sizeof(float),cudaMemcpyHostToDevice);
	if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
	  printf("Error in dev_MatrixA memory allocation:\nstatus0: %s, status1: %s\nExiting...\n\n",
  			cudaGetErrorString(cudastatus0),
			cudaGetErrorString(cudastatus1));
	  if(MatrixA)free(MatrixA);
	  if(MatrixB)free(MatrixB);
	  if(returnMatrix)free(returnMatrix);
	  if(dev_MatrixA) cudaFree(dev_MatrixA);
	  exit(EXIT_FAILURE);
	}


	float* dev_MatrixB;
	cudastatus2=cudaMalloc((void**)&dev_MatrixB,MATRIX_DIM_N*MATRIX_DIM_O*sizeof(float));
	cudastatus3=cudaMemcpy(dev_MatrixB,MatrixB,MATRIX_DIM_N*MATRIX_DIM_O*sizeof(float),cudaMemcpyHostToDevice);
	if(cudastatus2!=cudaSuccess|cudastatus3!=cudaSuccess){
	  printf("Error in dev_MatrixB memory allocation:\nstatus2: %s, status3: %s.\nExiting...\n\n",
  			cudaGetErrorString(cudastatus2),
			cudaGetErrorString(cudastatus3));
	  if(MatrixA)free(MatrixA);
	  if(MatrixB)free(MatrixB);
  	  if(returnMatrix)free(returnMatrix);
	  if(dev_MatrixA) cudaFree(dev_MatrixA);
	  if(dev_MatrixB) cudaFree(dev_MatrixB);
	  exit(EXIT_FAILURE);
	}


	float* dev_returnMatrix;
	cudastatus4=cudaMalloc((void**)&dev_returnMatrix,MATRIX_DIM_M*MATRIX_DIM_O*sizeof(float));
	cudastatus5=cudaMemset(dev_returnMatrix,0.0,MATRIX_DIM_M*MATRIX_DIM_O*sizeof(float));
	if(cudastatus4!=cudaSuccess|cudastatus5!=cudaSuccess){
	  printf("Error in dev_returnMatrix memory allocation:\nstatus4: %s, status5: %s\nExiting...\n\n",
	  		cudaGetErrorString(cudastatus4),
			cudaGetErrorString(cudastatus5));
	  if(MatrixA)free(MatrixA);
	  if(MatrixB)free(MatrixB);
	  if(returnMatrix)free(returnMatrix);
	  if(dev_MatrixA) cudaFree(dev_MatrixA);
	  if(dev_MatrixB) cudaFree(dev_MatrixB);
	  if(dev_returnMatrix) cudaFree(dev_returnMatrix);
	  exit(EXIT_FAILURE);
	}


//	update constant memory
	unsigned int matrixdims[4]={MATRIX_DIM_M,MATRIX_DIM_N,MATRIX_DIM_O,0};
	cudastatus6=cudaMemcpyToSymbol("symbolDIMS",matrixdims,4*sizeof(unsigned int));
	if(cudastatus6!=cudaSuccess){
	  printf("Error in symbol copy:\nstatus6: %s.\nExiting...\n\n",
	  		cudaGetErrorString(cudastatus6));
	  if(MatrixA)free(MatrixA);
	  if(MatrixB)free(MatrixB);
	  if(returnMatrix)free(returnMatrix);
	  if(dev_MatrixA) cudaFree(dev_MatrixA);
	  if(dev_MatrixB) cudaFree(dev_MatrixB);
	  if(dev_returnMatrix) cudaFree(dev_returnMatrix);
	  exit(EXIT_FAILURE);
	}
	ce=getclock();

//	set thread grid layout
	printf("----------------------------------------------------------------\n\n");

	int blockX = ceil((float)MATRIX_DIM_M/(float)TILE_WIDTH);
	int blockY = ceil((float)MATRIX_DIM_O/(float)TILE_WIDTH);

	printf("blockX: %d, blockY: %d, TILE_WIDTH: %d\n",blockX,blockY,TILE_WIDTH);
	

//	start timer	
	cudaEventRecord(start,0);

//	call kernel
	if(blockX>blockY)
	  blockY=blockX;
	else
	  blockX=blockY;

	dim3 dimGrid(blockX,blockY);

	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
	multMM_kernel<<<dimGrid,dimBlock>>>(dev_MatrixA,dev_MatrixB,dev_returnMatrix);


//	end timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop); // barrier

//	kernel time
	cudaEventElapsedTime(&elapsedtime,start,stop);
	cudastatus7=cudaMemcpy(returnMatrix,dev_returnMatrix,
		MATRIX_DIM_M*MATRIX_DIM_O*sizeof(float),cudaMemcpyDeviceToHost);

	if(cudastatus7!=cudaSuccess){
          printf("Error, kernel return status: %s\nExiting...\n\n",
	  		cudaGetErrorString(cudastatus7));
	  if(MatrixA)free(MatrixA);
	  if(MatrixB)free(MatrixB);
	  if(returnMatrix)free(returnMatrix);
	  if(dev_MatrixA) cudaFree(dev_MatrixA);
	  if(dev_MatrixB) cudaFree(dev_MatrixB);
	  if(dev_returnMatrix) cudaFree(dev_returnMatrix);
          exit(EXIT_FAILURE);
        }else{
	  printf("Kernel returned SUCCESS, elapsed time: %6.9lf sec.\n",elapsedtime/1000.0);
	  printf("Data set up time, elapsed time: %6.9lf sec.\n",ce-cs);
          printf("Total gpu kernel elapsed time: %6.9lf sec.\n",(elapsedtime/1000.0)+(ce-cs));
	  printf("................................................\n");
	}


/*
	for(i=0;i<MATRIX_DIM_M;i++){
		for(j=0;j<MATRIX_DIM_O;j++){
			printf("gpuMatrix[%d][%d]: %f\n",i,j,returnMatrix[i*MATRIX_DIM_O+j]);
		}
	}
*/


	printf("\nCalculating with openmp on CPU with 8 cores...");
	cs = getclock();
	float sum =0.;
	float* MatrixC = (float*) calloc(MATRIX_DIM_M*MATRIX_DIM_O,sizeof(float));
	#pragma omp parallel for private(i,j,k,sum)
	for(i=0;i<MATRIX_DIM_M;i++){
		for(j=0;j<MATRIX_DIM_O;j++){
		    sum=0.;
		    for(k=0;k<MATRIX_DIM_N;k++)
		        sum+=MatrixA[i*MATRIX_DIM_N+k]*MatrixB[k*MATRIX_DIM_O+j];
		    MatrixC[i*MATRIX_DIM_O+j]=sum;
		}
		
	}
	ce = getclock();
	printf("finished, elapsed time: %6.9lf sec.\n",ce-cs);
	

	printf("Calculating on CPU in serial...");
	cs = getclock();
	for(i=0;i<MATRIX_DIM_M;i++){
		for(j=0;j<MATRIX_DIM_O;j++){
		    sum=0.;
		    for(k=0;k<MATRIX_DIM_N;k++)
		        sum+=MatrixA[i*MATRIX_DIM_N+k]*MatrixB[k*MATRIX_DIM_O+j];
		    MatrixC[i*MATRIX_DIM_O+j]=sum;
		}
	}
	ce = getclock();
	printf("finished, elapsed time: %6.9lf sec.\n",ce-cs);


	float answer=0.;
	unsigned int counter=0;
	printf("\n\nVerifying correct answer from gpu.\n");
	for(i=0;i<MATRIX_DIM_M*MATRIX_DIM_O;i++){
		answer=MatrixC[i]-returnMatrix[i];
	//	printf("answer: %f\n",answer);
	//	printf("CPUmatrix: %f\n",MatrixC[i]);
	//	printf("GPUmatrix: %f\n",returnMatrix[i]);
		if (answer>0.00001){	
		   printf("Error, divergent value: %0.9f for index: %d\n",answer,i);
		   printf("CPUmatrix element (serial): %6.9f\n",MatrixC[i]);
		   printf("GPUmatrix element: %6.9f\n",returnMatrix[i]);
		   counter++;
		   if(counter>200) break;
		}
	}



	/*printf("Writing out to file...\n");
	FILE* fp=fopen(outfile,"w");
	for(i=0;i<MATRIX_DIM;i++){
	    fprintf(fp,"%f ",y[i]);
	}
	fclose(fp);*/

	printf("Freeing memory...\n");
	if(MatrixA)free(MatrixA);
	if(MatrixB)free(MatrixB);
	if(MatrixC)free(MatrixC);
	if(returnMatrix)free(returnMatrix);
	if(dev_MatrixA) cudaFree(dev_MatrixA);
	if(dev_MatrixB) cudaFree(dev_MatrixB);
	if(dev_returnMatrix) cudaFree(dev_returnMatrix);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Done, exiting...\n\n\n");
	
	exit(EXIT_SUCCESS);
}






__global__ void multMM_kernel(float* A, float* B, float* C){
	

	// symbolDIMS[0]==> M
	// symbolDIMS[1]==> N inner (width)
	// symbolDIMS[2]==> O

	unsigned int i,j;

	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

	unsigned int Arow = blockIdx.y*TILE_WIDTH + threadIdx.y;
	unsigned int Acolumn, Brow;
	unsigned int Bcolumn = blockIdx.x*TILE_WIDTH + threadIdx.x;
	unsigned int Imax = ceilf((float)symbolDIMS[1]/(float)TILE_WIDTH);
	//unsigned int Amax = symbolDIMS[0]*symbolDIMS[1];
	//unsigned int Bmax = symbolDIMS[1]*symbolDIMS[2];
	//unsigned int Cmax = symbolDIMS[0]*symbolDIMS[2];
	unsigned int Crow, Ccolumn;

	float value = 0.;

	for(i=0;i<Imax;i++){
	   Acolumn = (i*TILE_WIDTH+threadIdx.x);
	   if(Arow < symbolDIMS[0] && Acolumn < symbolDIMS[1]){
   	      As[threadIdx.y][threadIdx.x]=A[Arow*symbolDIMS[1]+Acolumn];
	   }else{
	      As[threadIdx.y][threadIdx.x]=0.;
	   }

	   Brow = (i*TILE_WIDTH+threadIdx.y);
	   if(Brow < symbolDIMS[1] && Bcolumn < symbolDIMS[2]){
	      Bs[threadIdx.y][threadIdx.x]=B[Brow*symbolDIMS[2]+Bcolumn];
	   }else{
	      Bs[threadIdx.y][threadIdx.x]=0.;	
	   }
	
	   __syncthreads();	
	   
	   for(j=0;j<TILE_WIDTH;j++){
	      value+=As[threadIdx.y][j]*Bs[j][threadIdx.x];
	   }
	   __syncthreads();
	}
	
	Crow=Arow;
	Ccolumn=Bcolumn;
	if(Crow < symbolDIMS[0] && Ccolumn < symbolDIMS[2]) 
	   C[Crow*symbolDIMS[2]+Ccolumn]=value;

}//end function





double getclock(){

      struct timezone tzp;
      struct timeval tp;
      gettimeofday (&tp, &tzp);
      return (tp.tv_sec + tp.tv_usec*1.0e-6);

}











