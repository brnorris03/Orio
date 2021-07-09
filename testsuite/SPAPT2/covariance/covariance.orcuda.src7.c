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

    param TC2[]  = range(32,1025,32);	# threads per block
    param BC2[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
    param ILUF2[] = range(1,33);
    param CB2[] = [False, True];		
    param PL2[] = [16,32,48];

    param TC3[]  = range(32,1025,32);	# threads per block
    param BC3[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
    param ILUF3[] = range(1,33);
    param CB3[] = [False, True];		
    param PL3[] = [16,32,48];

    param CFLAGS[] = ['-O3','-use_fast_math'];
  }
  
  def search {
    arg algorithm = 'Randomlocal';
    arg total_runs = 10000;
  }

  def input_params {
    param M = 3000; 
    param N = 500;
  }

  def input_vars {
    decl int m = M;
    decl int n = N;
    decl static double data[(M+10) * (N+10)] = random;
    decl static double symmat[(M+10) * (M+10)] = random;
    decl static double mean[M+10] = random;
    decl double float_n = 321414134.01;
    decl double eps = 0.005;
  }


) @*/   


#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(

  transform CUDA(threadCount=TC1, blockCount=BC1, cacheBlocks=CB1, preferL1Size=PL1, unrollInner=ILUF1)
  for (j = 1; j <= m; j++)
    {
      mean[j] = 0.0;
      for (i = 1; i <= n; i++)
        mean[j] += data[i*m+j];
      mean[j] /= float_n;
    }


  transform CUDA(threadCount=TC2, blockCount=BC2, cacheBlocks=CB2, preferL1Size=PL2, unrollInner=ILUF2)
  for (i = 1; i <= n; i++)
    for (j = 1; j <= m; j++)
      data[i*m+j] -= mean[j];

  transform CUDA(threadCount=TC3, blockCount=BC3, cacheBlocks=CB3, preferL1Size=PL3, unrollInner=ILUF3)
  for (k = 1; k <= m-1; k++) {
    symmat[k*m +k] = 1.0;
    for (j = k+1; j <= m; j++) {
      symmat[k*m+j] = 0.0;
      for (i = 1; i <= n; i++)
        symmat[k*m+j] += (data[i*m+k] * data[i*m+j]);
      symmat[j*m+k] = symmat[k*m+j];
    }
  }


) @*/

/*@ end @*/
/*@ end @*/
