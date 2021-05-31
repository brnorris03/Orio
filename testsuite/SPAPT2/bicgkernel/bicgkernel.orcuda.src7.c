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

    param CFLAGS[] = ['-O3','-use_fast_math'];
  }

  def search { 
    arg algorithm = 'Randomlocal'; 
    arg total_runs = 10000;
  } 
  
  def input_params {
    param NX[] = [10000];
    param NY[] = [10000];
  }
  
  def input_vars {
    decl int nx = NX;
    decl int ny = NY;
    decl static double A[NX*NY] = 0;  # need this for orcuda
    decl static double r[NX] = 0;
    decl static double s[NY] = 0;
    decl static double p[NY] = 0;
    decl static double q[NX] = 0;
    arg init_file = 'init2.c';
  }

) @*/

#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

int i,j;

/*@ begin Loop(

  transform CUDA(threadCount=TC1, blockCount=BC1, cacheBlocks=CB1, preferL1Size=PL1, unrollInner=ILUF1)
  for (i = 0; i <= nx-1; i++) 
    for (j = 0; j <= ny-1; j++) {
      s[j] = s[j] + r[i]*A[i*ny+j];
      q[i] = q[i] + A[i*ny+j]*p[j];
    }

) @*/
/*@ end @*/

/*@ end @*/
  


