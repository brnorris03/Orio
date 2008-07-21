#
# The class for storing information about transformation arguments
#

import sys

#---------------------------------------------------------------------

class ArgInfo:
    '''Transformation argument information'''

    # SIMD options
    SIMD_NONE = 'none'
    SIMD_GCC = 'gcc'
    SIMD_XLC = 'xlc'

    # block structure options
    BSTRUC_NONE = 'none'
    BSTRUC_INODE = 'inode'
    BSTRUC_BCSR = 'bcsr'

    #---------------------------------------------------------------------

    def __init__(self, out_vector, in_vector, in_matrix, row_inds, col_inds, data_type, init_val,
                 total_rows, total_inodes, inode_row_sizes, out_unroll_factor, in_unroll_factor,
                 num_threads, simd, block_structure):
        '''To instantiate transformation argument information'''

        self.out_vector = out_vector
        self.in_vector = in_vector
        self.in_matrix = in_matrix
        self.row_inds = row_inds
        self.col_inds = col_inds
        self.data_type = data_type
        self.init_val = init_val
        self.total_rows = total_rows
        self.total_inodes = total_inodes
        self.inode_row_sizes = inode_row_sizes

        self.out_unroll_factor = out_unroll_factor
        self.in_unroll_factor = in_unroll_factor
        self.num_threads = num_threads
        self.simd = simd
        self.block_structure = block_structure
        
        # check for unknown SpMV option
        if self.simd not in (self.SIMD_NONE, self.SIMD_GCC, self.SIMD_XLC):
            print 'error:SpMV: unknown SIMD value. got: "%s"' % self.simd
            sys.exit(1)

        # check the semantic correctness
        if self.simd != self.SIMD_NONE and self.in_unroll_factor % 2 != 0:
            print 'error:SpMV: inner unroll factor must be divisible by 2 for simdization'
            sys.exit(1)

        # check for unknown block-structure option
        if self.block_structure not in (self.BSTRUC_NONE, self.BSTRUC_INODE, self.BSTRUC_BCSR):
            print 'error:SpMV: unknown block-structure value. got: "%s"' % self.block_structure
            sys.exit(1)

#---------------------------------------------------------------------
    
class ArgInfoGen:
    '''A generator for the transformation argument information '''

    def __init__(self):
        '''To instantiate a generator for the transformation argument information'''
        pass

    #---------------------------------------------------------------------

    def generate(self, args, perf_params):
        '''To generate the transformation argument information'''

        # expected argument names
        OVEC = 'out_vector'
        IVEC = 'in_vector'
        IMAT = 'in_matrix'
        RINDS = 'row_inds'
        CINDS = 'col_inds'
        DTYPE = 'data_type'
        INITVAL = 'init_val'
        TROWS = 'total_rows'
        TINODES = 'total_inodes'
        IRSIZES = 'inode_row_sizes'
        OUFAC = 'out_unroll_factor'
        IUFAC = 'in_unroll_factor'
        NTHREADS = 'num_threads'
        SIMD = 'simd'
        BSTRUC = 'block_structure'

        # argument information
        out_vector = 'y'
        in_vector = 'x'
        in_matrix = 'aa'
        row_inds = 'ai'
        col_inds = 'aj'
        data_type = 'double'
        init_val = '0.0'
        total_rows = 'total_rows'
        total_inodes = 'total_inodes'
        inode_row_sizes = 'inode_row_sizes'
        out_unroll_factor = '1'
        in_unroll_factor = '1'
        num_threads = '1'
        simd = 'none'
        block_structure = 'none'

        # check that no argument names are repeated
        vnames = {}
        for _, (vname, line_no), (_, _) in args:
            if vname in vnames:
                print 'error:SpMV:%s: argument name "%s" already defined' % (line_no, vname)
                sys.exit(1)
            vnames[vname] = None
                                                                    
        # iterate over each argument
        for line_no, (vname, vname_line_no), (rhs, rhs_line_no) in args:
            
            # get argument value
            if vname == OVEC:
                out_vector = rhs
            elif vname == IVEC:
                in_vector = rhs
            elif vname == IMAT:
                in_matrix = rhs
            elif vname == RINDS:
                row_inds = rhs
            elif vname == CINDS:
                col_inds = rhs
            elif vname == DTYPE:
                elm_type = rhs
            elif vname == INITVAL:
                init_val = rhs
            elif vname == TROWS:
                total_rows = rhs
            elif vname == TINODES:
                total_inodes = rhs
            elif vname == IRSIZES:
                inode_row_sizes = rhs
            elif vname == OUFAC:
                out_unroll_factor = rhs
            elif vname == IUFAC:
                in_unroll_factor = rhs
            elif vname == NTHREADS:
                num_threads = rhs
            elif vname == SIMD:
                simd = rhs
            elif vname == BSTRUC:
                block_structure = rhs

            # unrecognized argument names
            else:
                print 'error:SpMV:%s: unrecognized argument name: "%s"' % (vname_line_no, vname)
                sys.exit(1)

        # evaluate the unroll factor values
        for name, rhs in (('unroll factor of the outer loop', out_unroll_factor),
                          ('unroll factor of the inner loop', in_unroll_factor),
                          ('number of threads', num_threads)):
            try:
                rhs_val = eval(rhs, perf_params)
            except Exception, e:
                print 'error:SpMV: failed to evaluate the RHS expression: %s' % rhs
                print ' --> %s: %s' % (e.__class__.__name__, e)
                sys.exit(1)
            if not isinstance(rhs_val, int) or rhs_val <= 0:
                print 'error:SpMV: %s must be a positive integer. got: %s' % (name, rhs_val)
                sys.exit(1)
        out_unroll_factor = eval(out_unroll_factor, perf_params)
        in_unroll_factor = eval(in_unroll_factor, perf_params)
        num_threads = eval(num_threads, perf_params)

        # evaluate the simd value
        for name, rhs in (('simdization', simd),
                          ('block structure', block_structure)):
            try:
                rhs_val = eval(rhs, perf_params)
            except Exception, e:
                print 'error:SpMV: failed to evaluate the RHS expression: %s' % rhs
                print ' --> %s: %s' % (e.__class__.__name__, e)
                sys.exit(1)
            if not isinstance(rhs_val, str):
                print 'error:SpMV: value of %s must be a string. got: %s' % (name, rhs_val)
                sys.exit(1)
        simd = eval(simd, perf_params)
        block_structure = eval(block_structure, perf_params)

        # generate and return the transformation argument information
        return ArgInfo(out_vector, in_vector, in_matrix, row_inds, col_inds, data_type, init_val,
                       total_rows, total_inodes, inode_row_sizes, out_unroll_factor, in_unroll_factor,
                       num_threads, simd, block_structure)
    
    
