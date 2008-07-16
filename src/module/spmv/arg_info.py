#
# The class for storing information about input arguments
#

import sys

#---------------------------------------------------------------------

class ArgInfo:
    '''Input argument information'''

    # SpMV options
    DEFAULT = 'DEFAULT'   # general assumption: compressed sparse row matrix
    INODE = 'INODE'       # assumption: use inode format, where rows have identical nonzero structure

    #---------------------------------------------------------------------

    def __init__(self, option, num_rows, num_cols, out_vector, in_vector, in_matrix, row_inds,
                 col_inds, out_loop_var, in_loop_var, elm_type, init_val,
                 out_unroll_factor, in_unroll_factor):
        '''To instantiate input argument information'''

        self.option = option
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.out_vector = out_vector
        self.in_vector = in_vector
        self.in_matrix = in_matrix
        self.row_inds = row_inds
        self.col_inds = col_inds
        self.out_loop_var = out_loop_var
        self.in_loop_var = in_loop_var
        self.elm_type = elm_type
        self.init_val = init_val
        
        self.out_unroll_factor = out_unroll_factor
        self.in_unroll_factor = in_unroll_factor

        # check for unknown SpMV option
        if self.option not in (self.DEFAULT, self.INODE):
            print 'error:SpMV: unknown SpMV option. got: %s' % self.option
            sys.exit(1)

    #---------------------------------------------------------------------

    def __str__(self):
        '''Return a string representation for this instance'''
        return repr(self)
    
    def __repr__(self):
        '''Return a string representation for this instance'''

        s = ''
        s += '-------------------------------\n'
        s += ' Argument information for SpMV  \n'
        s += '-------------------------------\n'
        s += '  option = %s \n' % self.option
        s += '  num_rows = %s \n' % self.num_rows
        s += '  num_cols = %s \n' % self.num_cols
        s += '  out_vector = %s \n' % self.out_vector
        s += '  in_vector = %s \n' % self.in_vector
        s += '  in_matrix = %s \n' % self.in_matrix
        s += '  row_inds = %s \n' % self.row_inds
        s += '  col_inds = %s \n' % self.col_inds
        s += '  out_loop_var = %s \n' % self.out_loop_var
        s += '  in_loop_var = %s \n' % self.in_loop_var
        s += '  elm_type = %s \n' % self.elm_type
        s += '  init_val = %s \n' % self.init_val
        s += '  out_unroll_factor = %s \n' % self.out_unroll_factor
        s += '  in_unroll_factor = %s \n' % self.in_unroll_factor
        return s
        
#---------------------------------------------------------------------
    
class ArgInfoGen:
    '''A generator for the input argument information '''

    def __init__(self):
        '''To instantiate a generator for the input argument information'''
        pass

    #---------------------------------------------------------------------

    def generate(self, args, perf_params):
        '''To generate the input argument information'''

        # expected argument names
        OPT = 'option'
        NROWS = 'num_rows'
        NCOLS = 'num_cols'
        OVEC = 'out_vector'
        IVEC = 'in_vector'
        IMAT = 'in_matrix'
        RINDS = 'row_inds'
        CINDS = 'col_inds'
        OLVAR = 'out_loop_var'
        ILVAR = 'in_loop_var'
        ETYPE = 'elm_type'
        INITVAL = 'init_val'
        OUFAC = 'out_unroll_factor'
        IUFAC = 'in_unroll_factor'

        # argument information
        option = 'DEFAULT'
        num_rows = None
        num_cols = None
        out_vector = None
        in_vector = None
        in_matrix = None
        row_inds = None
        col_inds = None
        out_loop_var = None
        in_loop_var = None
        elm_type = 'double'
        init_val = None
        out_unroll_factor = '1'
        in_unroll_factor = '1'

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
            if vname == OPT:
                option = rhs
            elif vname == NROWS:
                num_rows = rhs
            elif vname == NCOLS:
                num_cols = rhs
            elif vname == OVEC:
                out_vector = rhs
            elif vname == IVEC:
                in_vector = rhs
            elif vname == IMAT:
                in_matrix = rhs
            elif vname == RINDS:
                row_inds = rhs
            elif vname == CINDS:
                col_inds = rhs
            elif vname == OLVAR:
                out_loop_var = rhs
            elif vname == ILVAR:
                in_loop_var = rhs
            elif vname == ETYPE:
                elm_type = rhs
            elif vname == INITVAL:
                init_val = rhs
            elif vname == OUFAC:
                out_unroll_factor = rhs
            elif vname == IUFAC:
                in_unroll_factor = rhs

            # unrecognized argument names
            else:
                print 'error:SpMV:%s: unrecognized argument name: "%s"' % (vname_line_no, vname)
                sys.exit(1)

        # check some argument information values
        for rhs in (out_unroll_factor, in_unroll_factor):
            try:
                rhs_val = eval(rhs, perf_params)
            except Exception, e:
                print 'error:SpMV: failed to evaluate the RHS expression: %s' % rhs
                print ' --> %s: %s' % (e.__class__.__name__, e)
                sys.exit(1)
            if not isinstance(rhs_val, int) or rhs_val <= 0:
                print 'error:SpMV: unroll factor must be a positive integer. got: %s' % rhs_val
                sys.exit(1)

        # evaluate some argument information values
        out_unroll_factor = eval(out_unroll_factor, perf_params)
        in_unroll_factor = eval(in_unroll_factor, perf_params)

        # list of all expected argument names
        arg_names = [OPT, NROWS, NCOLS, OVEC, IVEC, IMAT, RINDS, CINDS, OLVAR, ILVAR, ETYPE,
                     INITVAL, OUFAC, IUFAC]

        # list of all argument information
        arg_infos = [option, num_rows, num_cols, out_vector, in_vector, in_matrix, row_inds,
                     col_inds, out_loop_var, in_loop_var, elm_type, init_val, out_unroll_factor,
                     in_unroll_factor]
        
        # check for undefined arguments
        if None in arg_infos:
            ipos = arg_infos.index(None)
            print 'error:SpMV: missing argument: "%s"' % arg_names[ipos]
            sys.exit(1)
        
        # generate and return the input argument information
        return ArgInfo(option, num_rows, num_cols, out_vector, in_vector, in_matrix, row_inds,
                       col_inds, out_loop_var, in_loop_var, elm_type, init_val,
                       out_unroll_factor, in_unroll_factor)




