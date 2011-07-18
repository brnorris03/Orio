#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>


#define NTYPES  3
#define MAXVARS 16



// Array counts by type
static int dcount=0;
static int fcount=0;
static int icount=0;


// CPU memory
double *dvec[MAXVARS];
float *fvec[MAXVARS];
int *ivec[MAXVARS];


// Host memory
double* devdVec[MAXVARS];
float* devfVec[MAXVARS];
int* deviVec[MAXVARS];



static dim3 ngrids;
static dim3 nblocks;
static dim3 nthreads;







void freeAllmem(){

     int i;

     printf("Freeing integer variables\n");
     for(i=0;i<=icount;i++){
         if(ivec[icount])free(ivec[icount]);
         if(deviVec[icount])free(deviVec[icount]);
     }

     printf("Freeing float variables\n");
     for(i=0;i<=fcount;i++){
         if(fvec[fcount])free(fvec[fcount]);
         if(devfVec[fcount])free(devfVec[fcount]);
     }

     printf("Freeing double variables\n");
     for(i=0;i<=dcount;i++){
	 if(dvec[dcount])free(dvec[dcount]);
	 if(devdVec[dcount])free(devdVec[dcount]);
     }

     printf("Done freeing memory...\n\n\n");
     return;
}




void init_kernel(int argc,...){
	
	


	int i;

	//Get device information
	int count;
	cudaGetDeviceCount(&count);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	printf("\n\nCompute capability: %d.%d\n",prop.major,prop.minor);


	cudaError_t cudastatus0,cudastatus1;

	int arraysize;
	int typesizes[NTYPES]={sizeof(double),sizeof(float),sizeof(int)};
	int tnum=-1;
	
	// stdarg list vars
	int vindex;
	va_list	varlist;
	va_start(varlist,argc);


	for(vindex=0;vindex<argc/2;vindex++){
	   
	   tnum=va_arg(varlist,int);

	   if(tnum==0){
	   	   printf("......found a double.\n");
		   arraysize=va_arg(varlist,int);

		   dvec[dcount] = (double*) calloc(arraysize, typesizes[tnum]);

		   cudastatus0=cudaMalloc((void**)&devdVec[dcount],arraysize*typesizes[tnum]);
		   cudastatus1=cudaMemcpy(devdVec[dcount],dvec[dcount],arraysize*typesizes[tnum],
				cudaMemcpyHostToDevice);

		   // Error check
		   if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
			printf("Error in devdVec memory allocation:\nstatus0: %s\nstatus1: %s\n",
  				cudaGetErrorString(cudastatus0),
			        cudaGetErrorString(cudastatus1));
	 
			freeAllmem();
          	   	exit(EXIT_FAILURE);
	           }

		   // update double vector count
		   dcount++;
     		   printf("Allocated double variable of size: %d\n",arraysize);

	   }else if(tnum==1){
	   	   printf("......found a float.\n");
		   arraysize=va_arg(varlist,int);

		   fvec[fcount] = (float*) calloc(arraysize, typesizes[tnum]);

		   cudastatus0=cudaMalloc((void**)&devfVec[fcount],arraysize*typesizes[tnum]);
		   cudastatus1=cudaMemcpy(devfVec[fcount],fvec[fcount],arraysize*typesizes[tnum],
				cudaMemcpyHostToDevice);

		   // Error check
		   if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
			printf("Error in devfVec memory allocation:\nstatus0: %s\nstatus1: %s\n",
  				cudaGetErrorString(cudastatus0),
			        cudaGetErrorString(cudastatus1));

			freeAllmem();
          	   	exit(EXIT_FAILURE);
	           }
		   
		   // update float vector count
		   fcount++;
     		   printf("Allocated float variable of size: %d\n",arraysize);

	   }else if(tnum==2){
	   	   printf("......found an int.\n");
		   arraysize=va_arg(varlist,int);

		   ivec[icount] = (int*) calloc(arraysize, typesizes[tnum]);

		   cudastatus0=cudaMalloc((void**)&deviVec[icount],arraysize*typesizes[tnum]);
		   cudastatus1=cudaMemcpy(deviVec[icount],ivec[icount],arraysize*typesizes[tnum],
				cudaMemcpyHostToDevice);

		   // Error check
		   if(cudastatus0!=cudaSuccess|cudastatus1!=cudaSuccess){
			printf("Error in deviVec memory allocation:\nstatus0: %s\nstatus1: %s\n",
  				cudaGetErrorString(cudastatus0),
			        cudaGetErrorString(cudastatus1));

 			freeAllmem();
          	   	exit(EXIT_FAILURE);
	           }
		   
		   // update float vector count
		   icount++;
     		   printf("Allocated integer variable of size: %d\n",arraysize);

	   }else{
		   printf("Error, unknown datatype: %d.\n", tnum);
		   freeAllmem();
		   printf("Exiting on failure...\n\n\n");
		   exit(EXIT_FAILURE);
	   }

		    
	}//end for-loop

	va_end(varlist);
	

	return;

}

