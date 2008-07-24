
/*@ begin SpMV (

 # SpMV computation: y = y + aa * x;
 out_vector = y;
 in_vector = x;
 in_matrix = aa;
 row_inds = ai;
 col_inds = aj;
 data_type = PetscScalar;
 init_val = 0;
 total_rows = total_rows;
 total_inodes = total_inodes;
 inode_sizes = inode_sizes;
 inode_rows = inode_rows;

 # transformation parameters
 out_unroll_factor = 4;
 in_unroll_factor = 4;
 num_threads = 1;
 simd = 'sse';    # 'none' (default), 'gcc', 'sse', 'xlc' (still unsupported)
 block_structure = 'none';    # 'none' (default), 'inode', 'bcsr' (still unsupported)

) @*/

/*@ end @*/

