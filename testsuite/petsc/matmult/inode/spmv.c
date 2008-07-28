/*@ begin PerfTuning (  
 def build { 
   arg build_command = 'gcc -O3 -I/disks/fast/papi/include -L/disks/fast/papi/lib -lpapi';
 }

 def performance_counter {
   arg repetitions = 100;
 }

 def performance_params {
   param OUF[] = [1];
   param IUF[] = [4];
 }

 def input_params {
   param TROWS[] = [100000];
   param TCOLS[] = [100000];
   param BROWS[] = [4];
   param BCOLS[] = [20];
 }
 
 def input_vars { 
   arg decl_file = 'decl_code.h';
   arg init_file = 'init_code.c'; 
 } 
 
 def performance_test_code { 
   arg skeleton_code_file = 'skeleton_code.c';  
 } 
) @*/ 

register int clength,rlength;
register int n=node_max;
while (n--) {
  rlength = node_sizes[0];
  node_sizes++;
  clength=ai[1]-ai[0];
  ai+=rlength;
  
  /*@ begin SpMV (     
    # SpMV option 
    option = INODE; 
    
    # high-level description of the SpMV computation 
    num_rows = rlength; 
    num_cols = clength; 
    out_vector = y; 
    in_vector = x; 
    in_matrix = aa; 
    row_inds = ai; 
    col_inds = aj; 
    out_loop_var = i; 
    in_loop_var = j;
    elm_type = double;
    init_val = 0.0;
    
    # transformation parameters 
    out_unroll_factor = OUF; 
    in_unroll_factor = IUF; 
    
  ) @*/ 
  /*@ end @*/
 }

/*@ end @*/

