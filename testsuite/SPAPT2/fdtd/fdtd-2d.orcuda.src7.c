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
    param N = 2000;
    param tmax = 100;
  }

  def input_vars {
    decl int nx = N;
    decl int ny = N;
    decl static double ex[N * N] = random;
    decl static double ey[N * N] = random;
    decl static double hz[N * N] = random;
  }


) @*/   

int j,t;

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop (

  for(t=0; t<=tmax-1; t++) {

    for (j=0; j<=ny-1; j++)
      ey[0*t + j] = t;

    transform CUDA(threadCount=TC1, blockCount=BC1, cacheBlocks=CB1, preferL1Size=PL1, 
                   unrollInner=ILUF1, streamCount=SC1)
    for (i=1; i<=nx-1; i++)
      for (j=0; j<=ny-1; j++) 
        ey[i*ny + j] = ey[i*ny + j] - 0.5 * (hz[i*ny + j] - hz[(i-1)*ny + j]) 
                                    - 0.5 * (hz[i*ny + j] - hz[i*ny + j-1]);


    transform CUDA(threadCount=TC2, blockCount=BC2, cacheBlocks=CB2, preferL1Size=PL2, 
                   unrollInner=ILUF2, streamCount=SC2)
    for (i=0; i<=nx-1; i++)
      for (j=0; j<=ny-1; j++)
        hz[i*ny + j]=hz[i*ny + j]-0.7*(ex[i*ny + j+1] - ex[i*ny +j] + ey[(i+1)*ny + j] - ey[i*ny + j]);
}

) @*/
/*@ end @*/

/*@ end @*/
