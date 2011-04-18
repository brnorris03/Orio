
/*@ begin SpMV (

 # SpMV computation: y = y + aa * x;
 out_vector = y;
 in_vector = x;
 in_matrix = aa;
 row_inds = ai;
 col_inds = aj;
 data_type = PetscScalar;
 init_val = 0;
 total_rows = total_rows;       # the total number of rows of matrix 'aa'
 total_inodes = total_inodes;   # the total number of inodes
 inode_sizes = inode_sizes;     # the total number of rows for each inode (e.g. [2,3,4])
 inode_rows = inode_rows;       # the accumulated version of inode_sizes (e.g. [0,2,5,9])

 # transformation parameters
 out_unroll_factor = 4;
 in_unroll_factor = 6;
 num_threads = 4;
 simd = 'none';    # 'none' (default), 'gcc', 'sse', 'xlc'
 block_structure = 'none';    # 'none' (default), 'inode', 'bcsr' (still unsupported)

) @*/

/*@ end @*/

