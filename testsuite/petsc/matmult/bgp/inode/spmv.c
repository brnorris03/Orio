/*@ begin PerfTuning (  
 def build { 
   arg build_command = 'mpixlc_r -O3 -qsmp=omp:noauto -qstrict ';
   arg batch_command = 'qsub -n 256 -t 5 -q short';
   arg status_command = 'qstat';
   arg num_procs = 256;
 }

 def performance_counter {
   arg method = 'bgp counter';
   arg repetitions = 100;
 }

 def performance_params {
   param UNROLL_FAC_OUT[] = [1,2,3,4];
   param UNROLL_FAC_IN[] = [1,2,3,4,5,6,7,8];
   param N_THREADS[] = [2];
   param SIMD_TYPE[] = ['none','xlc'];
   param BLK_TYPE[] = ['none','inode'];

   constraint simd_unroll_factor = (SIMD_TYPE=='none' or UNROLL_FAC_IN%2==0);
 }

 def input_params {
   param G_NROWS[] = [74000];
   param G_NCOLS[] = [74000];
   param B_NROWS[] = [4];
   param B_NCOLS_MIN[] = [12];
   param B_NCOLS_MAX[] = [20];
   param B_NCOLS_STRIDE[] = [4];

   constraint square_x_y = (G_NROWS==G_NCOLS);
 }
 
 def input_vars { 
   arg decl_file = 'decl_code.h';
   arg init_file = 'init_code.c'; 
 } 
 
 def performance_test_code { 
   arg skeleton_code_file = 'skeleton_code.c';  
 } 

 def search
 {
   arg algorithm = 'Exhaustive';
 }  
 ) @*/ 

/*@ begin SpMV (
  # SpMV computation: y = y + aa * x;
  out_vector = y;
  in_vector = x;
  in_matrix = aa;
  row_inds = ai;
  col_inds = aj;
  data_type = double;
  init_val = 0;
  total_rows = total_rows;       
  total_inodes = total_inodes;   
  inode_sizes = inode_sizes;     
  inode_rows = inode_rows;       
  
  # transformation parameters
  out_unroll_factor = UNROLL_FAC_OUT;
  in_unroll_factor = UNROLL_FAC_IN;
  num_threads = N_THREADS;
  simd = SIMD_TYPE;              # 'none' (default), 'gcc', 'sse', 'xlc'
  block_structure = BLK_TYPE;    # 'none' (default), 'inode', 'bcsr' (still unsupported)
  ) @*/

/*@ end @*/

/*@ end @*/

