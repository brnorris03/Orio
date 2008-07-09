#
# The code generator for the SpMV transformation module
#

import re, sys
import arg_info

#-------------------------------------------

class CodeGen:
    '''The code generator for the SpMV transformation module'''

    def __init__(self, ainfo, annot_body_code, indent):
        '''To instantiate a code generator instance'''
        
        self.ainfo = ainfo
        self.annot_body_code = annot_body_code
        self.indent = indent

        # boolean values to adjust the code generation phase
        self.include_outer_loop = True
        self.include_cleanup_code = True
        if not self.include_outer_loop:
            self.include_cleanup_code = False

    #------------------------------------------------------

    def __generateCLengthCode(self):
        '''Generate code for computing the column length'''

        ainfo = self.ainfo
        clength_var = 'clength'
        clength_decl = 'short int %s;' % clength_var
        clength_init = '%s=%s[1]-%s[0];' % (clength_var, ainfo.row_inds, ainfo.row_inds)
        rind_inc = '%s+=%s;' % (ainfo.row_inds, ainfo.num_rows)
        return (clength_var, clength_decl, clength_init, rind_inc)
        
    #------------------------------------------------------

    def __generateIArrCode(self, clength_var):
        '''Generate code for the pointers to input array'''

        ainfo = self.ainfo
        iarrs = ['%s%s' % (ainfo.in_matrix, i) for i in range(0, ainfo.out_unroll_factor)]
        iarr_decls = ['%s *%s;' % (ainfo.elm_type, a) for a in iarrs]
        iarr_inits = [('%s=%s;' % (a, ainfo.in_matrix)) if i==0 else
                      ('%s=%s+%s;' % (a, iarrs[i-1], clength_var))
                      for i,a in enumerate(iarrs)]
        iarr_unroll_inc = '%s=%s;' % (ainfo.in_matrix, iarrs[-1])
        iarr_cleanup_inc = '%s=%s;' % (ainfo.in_matrix, iarrs[0])
        return (iarrs, iarr_decls, iarr_inits, iarr_unroll_inc, iarr_cleanup_inc)

    #------------------------------------------------------

    def __generateOVecCode(self):
        '''Generate code for the temporary variables used to keep scalar outputs in registers'''

        ainfo = self.ainfo
        ovecs = ['%s%s' % (ainfo.out_vector, i) for i in range(0, ainfo.out_unroll_factor)]
        orefs = ['%s[%s]' % (ainfo.out_vector, i) for i in range(0, ainfo.out_unroll_factor)]
        ovec_decls = ['%s %s;' % (ainfo.elm_type, v) for v in ovecs]
        ovec_inits = ['%s=%s;' % (v, ainfo.init_val) for v in ovecs]
        ovec_stores = ['%s=%s;' % (r,v) for r,v in zip(orefs, ovecs)]
        ovec_unroll_inc = '%s+=%s;' % (ainfo.out_vector, ainfo.out_unroll_factor)
        ovec_cleanup_inc = '%s++;' % (ainfo.out_vector)
        return (ovecs, ovec_decls, ovec_inits, ovec_stores, ovec_unroll_inc, ovec_cleanup_inc)

    #------------------------------------------------------

    def __generateIVecCode(self, clength_var):
        '''Generate code for storing values of the input vectors'''

        ainfo = self.ainfo
        cind_var = 'ind'
        cinds = ['%s%s' % (cind_var, i) for i in range(0, ainfo.in_unroll_factor)]
        cind_refs = ['%s[%s]' % (ainfo.col_inds, i) for i in range(0, ainfo.in_unroll_factor)]
        cind_decls = ['int %s;' % i for i in cinds]
        cind_inits = ['%s=%s;' % (i,r) for i,r in zip(cinds, cind_refs)]
        cind_unroll_inc = '%s+=%s;' % (ainfo.col_inds, ainfo.in_unroll_factor)
        cind_cleanup_inc = '%s++;' % (ainfo.col_inds)
        if ainfo.out_unroll_factor == 1:
            cind_inc = ''
        elif ainfo.out_unroll_factor == 2:
            cind_inc = '%s+=%s;' % (ainfo.col_inds, clength_var)
        else:
            cind_inc = '%s+=%s*%s;' % (ainfo.col_inds, (ainfo.out_unroll_factor-1), clength_var)
        ivecs = ['%s%s' % (ainfo.in_vector, i) for i in range(0, ainfo.in_unroll_factor)]
        ivec_decls = ['%s %s;' % (ainfo.elm_type, v) for v in ivecs]
        ivec_inits = ['%s=%s[%s];' % (v, ainfo.in_vector, i) for v,i in zip(ivecs, cinds)]
        return (cind_decls, cind_inits, cind_unroll_inc, cind_cleanup_inc, cind_inc,
                ivecs, ivec_decls, ivec_inits)

    #------------------------------------------------------

    def __generateOpCode(self, ovecs, iarrs, ivecs):
        '''Generate code for the unrolled SpMV operations'''

        ops = []
        for i1,(ov,a) in enumerate(zip(ovecs, iarrs)):
            for i2,iv in enumerate(ivecs):
                ops.append('%s = %s + %s[%s] * %s;' % (ov,ov,a,i2,iv))
        arr_incs = [('%s+=%s;' % (a, len(ivecs))) if len(ivecs)>1 else ('%s++;' % a) for a in iarrs]
        ops.append(' '.join(arr_incs))
        return ops
    
    #------------------------------------------------------

    def __generateInode(self):
        '''The code generation procedure for the inode case'''

        # get the argument info
        ainfo = self.ainfo
        
        # generate codes for computing the column length
        (clength_var, clength_decl, clength_init, rind_inc) = self.__generateCLengthCode()

        # generate codes for the input arrays
        (iarrs, iarr_decls, iarr_inits, iarr_unroll_inc,
         iarr_cleanup_inc) = self.__generateIArrCode(clength_var)

        # generate codes for the output vectors
        (ovecs, ovec_decls, ovec_inits, ovec_stores, ovec_unroll_inc,
         ovec_cleanup_inc) = self.__generateOVecCode()

        # generate codes for the input vectors
        (cind_decls, cind_inits, cind_unroll_inc, cind_cleanup_inc, cind_inc,
         ivecs, ivec_decls, ivec_inits) = self.__generateIVecCode(clength_var)

        # generate codes for the unrolled operation codes
        main_main_ops = self.__generateOpCode(ovecs, iarrs, ivecs)
        main_cleanup_ops = self.__generateOpCode(ovecs, iarrs, ivecs[:1])
        cleanup_main_ops = self.__generateOpCode(ovecs[:1], iarrs[:1], ivecs)
        cleanup_cleanup_ops = self.__generateOpCode(ovecs[:1], iarrs[:1], ivecs[:1])

        # generate code for the main unrolled loop
        code = '\n'
        code += clength_decl + '\n'
        code += clength_init + '\n'
        code += rind_inc + '\n'
        if self.include_outer_loop:
            code += 'for (%s=0; %s<=%s-%s; %s+=%s) {\n' % (ainfo.out_loop_var, ainfo.out_loop_var,
                                                           ainfo.num_rows,
                                                           (1 + ainfo.out_unroll_factor - 1),
                                                           ainfo.out_loop_var,
                                                           ainfo.out_unroll_factor)
        code += '  ' + ' '.join(iarr_decls) + '\n'
        code += '  ' + ' '.join(iarr_inits) + '\n'
        code += '  ' + ' '.join(ovec_decls) + '\n'
        code += '  ' + ' '.join(ovec_inits) + '\n'
        code += '  ' + 'for (%s=0; %s<=%s-%s; %s+=%s) {\n' % (ainfo.in_loop_var, ainfo.in_loop_var,
                                                              clength_var,
                                                              (1 + ainfo.in_unroll_factor - 1),
                                                              ainfo.in_loop_var,
                                                              ainfo.in_unroll_factor)
        code += '    ' + ' '.join(cind_decls) + '\n'
        code += '    ' + ' '.join(ivec_decls) + '\n'
        code += '    ' + ' '.join(cind_inits) + '\n'
        code += '    ' + cind_unroll_inc + '\n'
        code += '    ' + ' '.join(ivec_inits) + '\n'
        code += '    ' + '\n    '.join(main_main_ops) + '\n'
        code += '  ' + '} \n'
        code += '  ' + 'for (; %s<=%s-1; %s++) {\n' % (ainfo.in_loop_var, clength_var,
                                                       ainfo.in_loop_var)
        code += '    ' + cind_decls[0] + '\n'
        code += '    ' + ivec_decls[0] + '\n'
        code += '    ' + cind_inits[0] + '\n'
        code += '    ' + cind_cleanup_inc + '\n'
        code += '    ' + ivec_inits[0] + '\n'
        code += '    ' + '\n    '.join(main_cleanup_ops) + '\n'
        code += '  ' + '} \n'
        code += '  ' + ' '.join(ovec_stores) + '\n'
        code += '  ' + ovec_unroll_inc + '\n'
        code += '  ' + iarr_unroll_inc + '\n'
        code += '  ' + cind_inc + '\n'
        if self.include_outer_loop:
            code += '} \n'
        
        # generate code for the clean-up loop
        if self.include_cleanup_code:
            code += 'for (; %s<=%s-1; %s++) {\n' % (ainfo.out_loop_var, ainfo.num_rows,
                                                    ainfo.out_loop_var)
            code += '  ' + iarr_decls[0] + '\n'
            code += '  ' + iarr_inits[0] + '\n'
            code += '  ' + ovec_decls[0] + '\n'
            code += '  ' + ovec_inits[0] + '\n'
            code += '  ' + 'for (%s=0; %s<=%s-%s; %s+=%s) {\n' % (ainfo.in_loop_var,
                                                                  ainfo.in_loop_var,
                                                                  clength_var,
                                                                  (1 + ainfo.in_unroll_factor - 1),
                                                                  ainfo.in_loop_var,
                                                                  ainfo.in_unroll_factor)
            code += '    ' + ' '.join(cind_decls) + '\n'
            code += '    ' + ' '.join(ivec_decls) + '\n'
            code += '    ' + ' '.join(cind_inits) + '\n'
            code += '    ' + cind_unroll_inc + '\n'
            code += '    ' + ' '.join(ivec_inits) + '\n'
            code += '    ' + '\n    '.join(cleanup_main_ops) + '\n'
            code += '  ' + '} \n'
            code += '  ' + 'for (; %s<=%s-1; %s++) {\n' % (ainfo.in_loop_var, clength_var,
                                                           ainfo.in_loop_var)
            code += '    ' + cind_decls[0] + '\n'
            code += '    ' + ivec_decls[0] + '\n'
            code += '    ' + cind_inits[0] + '\n'
            code += '    ' + cind_cleanup_inc + '\n'
            code += '    ' + ivec_inits[0] + '\n'
            code += '    ' + '\n    '.join(cleanup_cleanup_ops) + '\n'
            code += '  ' + '} \n'
            code += '  ' + ovec_stores[0] + '\n'
            code += '  ' + ovec_cleanup_inc + '\n'
            code += '  ' + iarr_cleanup_inc + '\n'
            code += '} \n'

        # return the generated code
        return code

    #------------------------------------------------------

    def __generateDefault(self):
        '''The default code generation procedure'''

        print 'error: default code generation for SpMV is not yet implemented'
        sys.exit(1)
        
    #------------------------------------------------------

    def generate(self):
        '''To generate the optimized SpMV code'''

        # generate the optimized code
        if self.ainfo.option == arg_info.ArgInfo.INODE:
            code = self.__generateInode()
        else:
            code = self.__generateDefault()

        # return the code
        return code
    
