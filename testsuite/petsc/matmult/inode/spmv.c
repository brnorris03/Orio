/*@ begin PerfTuning (  
 def build { 
   arg command = 'gcc'; 
   arg options = -O -I/disks/fast/papi/include -L/disks/fast/papi/lib -lpapi';
 }

 def performance_counter {
   arg repetitions = 10;
 }

 def performance_params {
   param DUMMY[] = [1];
 }

 def input_params {
   param TROWS[] = [10000];
   param TCOLS[] = [10000];
   param BROWS[] = [4];
   param BCOLS[] = [16];
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
  rlength = row_sizes[0];
  ++row_sizes;
  clength=ai[1]-ai[0];
  ai+=rlength;
  
  /*@ begin SpMV (     
    # SpMV option 
    option = INODE; 
    
    # high-level description of the SpMV computation 
    num_rows = rlength; 
    out_vector = y; 
    in_vector = x; 
    in_matrix = aa; 
    row_inds = ai; 
    col_inds = aj; 
    out_loop_var = i; 
    in_loop_var = j;
    elm_type = double;
    init_val = 0;
    
    # transformation parameters 
    out_unroll_factor = 1; 
    in_unroll_factor = 1; 
    
  ) @*/ 
  /*@ end @*/
 }

/*@ end @*/

