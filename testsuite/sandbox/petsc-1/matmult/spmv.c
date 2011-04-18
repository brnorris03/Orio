
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
 out_unroll_factor = 2;
 in_unroll_factor = 4;
 
 ) @*/
/*@ end @*/
