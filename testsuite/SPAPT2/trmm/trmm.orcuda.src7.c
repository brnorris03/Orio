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
    param PL1[] = [16,32,48];
    param SC1[] = range(1,33);

    param TC2[]  = range(32,1025,32);	# threads per block
    param BC2[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
    param ILUF2[] = range(1,33);
    param CB2[] = [False, True];		
    param PL2[] = [16,32,48];
    param SC2[] = range(1,33);

    param CFLAGS[] = ['-O3','-use_fast_math'];
  }
  
  def search {
    arg algorithm = 'Randomlocal';
    arg total_runs = 10000;
  }

  def input_params {
    param N[] = [1000]; 
  }

  def input_vars {
    decl int n = N;
    decl double alpha = 1;
    decl static double A[N*N] = random;
    decl static double B[N*N] = random;
  }

) @*/   


#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(

  transform CUDA(threadCount=TC1, blockCount=BC1, cacheBlocks=CB1, preferL1Size=PL1, 
                   unrollInner=ILUF1, streamCount=SC1)
  for (i=0; i<=n-1; i++)
    for (j=0; j<=n-1; j++) {
      B[i*n+j] = alpha*A[i*n+i]*B[i*n+j];
      for (k=i+1; k<=n-1; k++)
	B[i*n+j] = B[i*n+j] + alpha*A[i*n+k]*B[k*n+j];
    }

  transform CUDA(threadCount=TC2, blockCount=BC2, cacheBlocks=CB2, preferL1Size=PL2, 
                 unrollInner=ILUF2, streamCount=SC2)
  for (i=0; i<=n-1; i++)
    for (j=0; j<=n-1; j++)
      for (k=i+1; k<=n-1; k++)
        B[i*n+j] = B[i*n+j] + alpha*A[i*n+k]*B[k*n+j];

) @*/

/*@ end @*/
/*@ end @*/
