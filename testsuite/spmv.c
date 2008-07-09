
/*@ begin SpMV(

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
  elm_type = PetscScalar;
  init_val = 0.0;

  # transformation parameters
  out_unroll_factor = 4;
  in_unroll_factor = 2;

) @*/

for (i=0; i<=m-1; i++) 
  { 
    y[i] = 0.0; 
    for (j=ai[i]; j<=ai[i+1]-1; j++) 
      y[i] = y[i] + aa[j] * x[aj[j]]; 
  } 


/*@ end @*/


