/*@ begin PerfTuning (        
  def build {
    arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
  }
   
  def performance_counter {
    arg repetitions = 3;
  }

  def performance_params {
    param thread_count[]  = range(32,1025,32);	# threads per block
    param block_count[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
    param inner_loop_unroll_fact[] = range(1,33);
    param cache_blocks[] = [False, True];		
    param preferred_L1_cache[]  = [16,32,48];
    param CFLAGS[] = ['-O3','-use_fast_math'];
  }

  def search {
    arg algorithm = 'Randomlocal';
    arg total_runs = 10000;
  }

  def input_params {
    param VSIZE = 500;
    param OSIZE = 25;
    param V = 500;
    param O = 25;
  }

  def input_vars {
    decl int v = V;
    decl int o = O;
    decl dynamic double A2[V*O] = random;
    decl dynamic double T[V*O*O*O] = random;
    decl dynamic double R[V*V*O*O] = 0;
  }

  def validation {
    arg validation_file = 'validation.c';
  }

) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))


/*@ begin Loop(
 transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
                 cacheBlocks=cache_blocks,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact)
  for(i=0; i<=v-1; i++) 
    for(j=0; j<=v-1; j++) 
      for(k=0; k<=o-1; k++) 
        for(l=0; l<=o-1; l++) 
	  for(m=0; m<=o-1; m++) 
	    R[i*V*O*O+j*O*O+k*O+l] = R[i*V*O*O+j*O*O+k*O+l] + T[i*O*O*O+m*O*O+k*O+l] * A2[j*O+m];

) @*/

/*@ end @*/
/*@ end @*/

