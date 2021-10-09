void ZGEMV( int nrows, int ncols, std::complex<double> alpha, std::complex<double>* A, std::complex<double>* X, std::complex<double> beta,  std::complex<double>* Y ){

    /* Y = alpha A X + beta Y */
    
/*@ begin PerfTuning (
    def performance_params {
        param thread_count[]  = range(32,1025,32);	# threads per block
        param block_count[]  = range(108,1081,108);	# grid.x; multiples of # SMs, 108 for A100
        param inner_loop_unroll_fact[] = range(1,33);
        param cache_blocks[] = [False, True];
        param preferred_L1_cache[]  = [16,32,48];
        param stream_count[] = range(1,31);
        param CFLAGS[] = ['-O3','-use_fast_math'];
    }

    def input_params {
        param M = 2048;
        param ALPHA = 1.0;
        param BETA = 2.0;
    }

    def input_vars {
        decl static doublecomplex A[M*M] = random;
        decl static doublecomplex X[M] = random;
        decl static doublecomplex Y[M] = random;
    }

    def search {
        arg algorithm = 'Mlsearch';
        arg total_runs = 100;
    }

    def build {
      arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
    }

    def performance_counter {
      arg method = 'basic timer';
      arg repetitions = 3;
    }
) @*/

//int n = M;
   int ncols = M;
   int nrows = M;
   /*   std::complex<double> alpha = ALPHA;
        std::complex<double> beta = BETA;*/
   double alpha = ALPHA;
   double beta = BETA;
    
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/*@ begin Loop(
  transform CUDA(threadCount=thread_count,
                 blockCount=block_count,
		         cacheBlocks=cache_blocks,
                 preferL1Size=preferred_L1_cache,
                 unrollInner=inner_loop_unroll_fact,
                 streamCount=stream_count)

  for(i=0; i <= nrows-1 ; i++){
   for(j=0; j <= ncols-1 ; j++) {
        Y[i] = alpha * X[i] * A[i*ncols+j] + beta * Y[i];
      }
    }
) @*/

/*@ end @*/
/*@ end @*/


}
