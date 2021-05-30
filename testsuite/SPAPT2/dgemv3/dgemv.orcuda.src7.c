/* void DGEMV(int A_nrows, int A_ncols, double A[A_ncols*A_nrows], int e_nrows, double e[e_nrows], int w_nrows, double w[w_nrows], int x_nrows, double x[x_nrows], int p_nrows, double p[p_nrows], int y_nrows, double y[y_nrows], int z_nrows, double z[z_nrows]) {
  int ii,i,j;
  double t2[A_nrows];
*/

/*@ begin PerfTuning (        
  def build {
    arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
  }
   
  def performance_counter {
    arg repetitions = 3;
  }
  
  def performance_params {
    param TC1[]  = range(32,1025,32);	# threads per block
    param BC1[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
    param ILUF1[] = range(1,33);
    param CB1[] = [False, True];		
    param PL1[]  = [16,32,48];

    param TC2[]  = range(32,1025,32);	# threads per block
    param BC2[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
    param CB2[] = [False, True];		
    param PL2[]  = [16,32,48];

    param TC3[]  = range(32,1025,32);	# threads per block
    param BC3[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
    param ILUF3[] = range(1,33);
    param CB3[] = [False, True];		
    param PL3[]  = [16,32,48];

    param CFLAGS[] = ['-O3','-use_fast_math'];
  } 
  
  def search {
    arg algorithm = 'Randomlocal';
    arg total_runs = 10000;
  }
  
  def input_params {
    param M = 1024;
    param N = 1024;
  } 
 
  def input_vars {
    decl int A_nrows = M;
    decl int A_ncols = N;
    decl static double A[M * N] = random;
    decl static double e[N] = random;
    decl static double x[N] = random;
    decl static double w[N] = random;
    decl static double y[M] = random;
    decl static double z[M] = random;
    decl static double p[M] = 0;
    decl static double t2[M] = 0;
    decl static double t6[M] = 0;
    decl static double t10[M] = 0;
  }            
) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


/*@ begin Loop (
  
   transform CUDA(threadCount=TC1, blockCount=BC1, cacheBlocks=CB1, preferL1Size=PL1, unrollInner=ILUF1)
   for (i = 0; i <= A_ncols-1; i++) 
     for (j = 0; j <= A_nrows-1; j++) 
       t2[j] = t2[j] + A[i*A_nrows + j] * x[i]; 

   transform CUDA(threadCount=TC2, blockCount=BC2, cacheBlocks=CB2, preferL1Size=PL2)
   for (i = 0; i <= A_nrows-1; i++) 
     y[i] = t2[i] + y[i]; 

   transform CUDA(threadCount=TC1, blockCount=BC1, cacheBlocks=CB1, preferL1Size=PL1, unrollInner=ILUF1)
   for (i = 0; i <= A_ncols-1; i++) 
     for (j = 0; j <= A_nrows-1; j++) 
       t6[j] = t6[j] + A[i*A_nrows + j] * w[i]; 

   transform CUDA(threadCount=TC2, blockCount=BC2, cacheBlocks=CB2, preferL1Size=PL2)
   for (i = 0; i <= A_nrows-1; i++) 
        z[i] = t6[i] + z[i]; 

   transform CUDA(threadCount=TC1, blockCount=BC1, cacheBlocks=CB1, preferL1Size=PL1, unrollInner=ILUF1)
   for (i = 0; i <= A_ncols-1; i++) 
     for (j = 0; j <= A_nrows-1; j++) 
       t10[j] = t10[j] + A[i*A_nrows + j] * e[i]; 

   transform CUDA(threadCount=TC2, blockCount=BC2, cacheBlocks=CB2, preferL1Size=PL2)
   for (i = 0; i <= A_nrows-1; i++) 
     p[i] = t10[i] + p[i]; 


   transform CUDA(threadCount=TC3, blockCount=BC3, cacheBlocks=CB3, preferL1Size=PL3, unrollInner=ILUF3)
   for (i = 0; i <= A_ncols-1; i++) 
     for (j = 0; j <= A_nrows-1; j++) {
       t10[j] = t10[j] + e[i] * A[i*A_nrows + j]; 
       t6[j] = t6[j] + w[i] * A[i*A_nrows + j]; 
       t2[j] = t2[j] + A[i*A_nrows + j] * x[i]; 
     }

   transform CUDA(threadCount=TC2, blockCount=BC2, cacheBlocks=CB2, preferL1Size=PL2)
   for (i = 0; i <= A_nrows-1; i++) {
     y[i] = t2[i] + y[i]; 
     z[i] = t6[i] + z[i]; 
     p[i] = t10[i] + p[i]; 
   }
) @*/

/* Noop */

/*@ end @*/

/*@ end @*/

//}
