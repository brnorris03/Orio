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
        self.vectorize = True

        # some option adjustments
        if not self.include_outer_loop:
            self.include_cleanup_code = False
        if self.ainfo.in_unroll_factor == 1:
            self.vectorize = False

        # check semantic correctness
        self.__semantCheck()

    #------------------------------------------------------

    def __semantCheck(self):
        '''Check semantic correctness'''

        # get the argument information
        ainfo = self.ainfo

        # semantic for vectorization
        if self.vectorize and ainfo.in_unroll_factor > 1 and ainfo.in_unroll_factor % 2 != 0:
            print ('error:SpMV: vectorization requires the inner loop unroll factor to be ' +
                   'divisible by two')
            sys.exit(1)

    #------------------------------------------------------

    def __generateIArrCode(self):
        '''Generate code for the pointers to input array'''

        ainfo = self.ainfo
        iarrs = ['%s%s' % (ainfo.in_matrix, i) for i in range(0, ainfo.out_unroll_factor)]
        iarr_decls = ['%s *%s;' % (ainfo.elm_type, a) for a in iarrs]
        iarr_inits = [('%s=%s;' % (a, ainfo.in_matrix)) if i==0 else
                      ('%s=%s+%s;' % (a, iarrs[i-1], ainfo.num_cols))
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

    def __generateIVecCode(self):
        '''Generate code for storing values of the input vectors'''

        ainfo = self.ainfo
        cind_unroll_inc = '%s+=%s;' % (ainfo.col_inds, ainfo.in_unroll_factor)
        cind_cleanup_inc = '%s++;' % (ainfo.col_inds)
        if ainfo.out_unroll_factor == 1:
            cind_inc = ''
        elif ainfo.out_unroll_factor == 2:
            cind_inc = '%s+=%s;' % (ainfo.col_inds, ainfo.num_cols)
        else:
            cind_inc = '%s+=%s*%s;' % (ainfo.col_inds, (ainfo.out_unroll_factor-1), ainfo.num_cols)
        ivecs = ['%s%s' % (ainfo.in_vector, i) for i in range(0, ainfo.in_unroll_factor)]
        ivec_decls = ['%s %s;' % (ainfo.elm_type, v) for v in ivecs]
        ivec_inits = ['%s=%s[%s[%s]];' % (v, ainfo.in_vector, ainfo.col_inds, i)
                      for i,v in enumerate(ivecs)]
        return (cind_unroll_inc, cind_cleanup_inc, cind_inc, ivecs, ivec_decls, ivec_inits)

    #------------------------------------------------------

    def __generateSimdCode(self, ovecs, ivecs, iarrs):
        '''Generate code for vectorization/simdization'''

        ainfo = self.ainfo
        simd_type = 'v2df'
        simd_typedef = 'typedef double %s __attribute__ ((vector_size(16)));' % simd_type
        simd_ovars = ['%s%so' % (ainfo.out_vector, i) for i in range(0, ainfo.out_unroll_factor)]
        simd_ovar_inits = ['%s %s={0,0};' % (simd_type, v) for v in simd_ovars]
        simd_ivecs = ['%s%sv' % (ainfo.in_vector, i) for i in range(0, ainfo.in_unroll_factor/2)]
        simd_ivec_inits = ['%s %s={%s[%s[%s]],%s[%s[%s]]};' % (simd_type,v,
                                                               ainfo.in_vector,ainfo.col_inds,i*2,
                                                               ainfo.in_vector,ainfo.col_inds,i*2+1)
                           for i,v in enumerate(simd_ivecs)]
        simd_iarrs = [['%s%sv' % (a,i) for i in range(ainfo.in_unroll_factor/2)] for a in iarrs]

        simd_iarr_inits = [' '.join(['%s %s%sv={%s[%s],%s[%s]};' % (simd_type,a,i,a,i*2,a,i*2+1)
                                     for i in range(0, ainfo.in_unroll_factor/2)])
                           for a in iarrs]
        simd_uvars = ['%s%su' % (ainfo.out_vector, i) for i in range(0, ainfo.out_unroll_factor)]
        simd_uvar_inits = ['double *%s=(double *)&%s;' % (u,o) for u,o in zip(simd_uvars, simd_ovars)]
        simd_stores = ['%s=%s[0]+%s[1];' % (o,u,u) for o,u in zip(ovecs, simd_uvars)]
        return (simd_typedef, simd_ovars, simd_ovar_inits, simd_ivecs, simd_ivec_inits,
                simd_iarrs, simd_iarr_inits, simd_uvar_inits, simd_stores)

    #------------------------------------------------------

    def __generateUnrolledOpCode(self, iarrs, inc_val, ovrefs, iarefs, ivrefs):
        '''Generate code for the unrolled SpMV computation code'''

        # initiate code sequence
        codes = []

        # generate the unrolled spmv computation code
        for ov,as in zip(ovrefs, iarefs):
            c = '%s +=' % ov
            for i,(a,iv) in enumerate(zip(as,ivrefs)):
                if i > 0:
                    c += ' +'
                c += ' %s*%s' % (a,iv)
            codes.append(c + ';')

        # generate code for incrementing the array pointers
        arr_incs = [('%s+=%s;' % (a, inc_val)) if inc_val>1 else ('%s++;' % a) for a in iarrs]
        codes.append(' '.join(arr_incs))

        # return the generated code
        return codes
    
    #------------------------------------------------------

    def __generateInode(self):
        '''The code generation procedure for the inode case'''

        # get the argument info
        ainfo = self.ainfo
        
        # generate codes for the input arrays
        (iarrs, iarr_decls, iarr_inits, iarr_unroll_inc,
         iarr_cleanup_inc) = self.__generateIArrCode()

        # generate codes for the output vectors
        (ovecs, ovec_decls, ovec_inits, ovec_stores, ovec_unroll_inc,
         ovec_cleanup_inc) = self.__generateOVecCode()

        # generate codes for the input vectors
        (cind_unroll_inc, cind_cleanup_inc, cind_inc,
         ivecs, ivec_decls, ivec_inits) = self.__generateIVecCode()

        # generate codes for vectorization/simdization
        if self.vectorize:
            (simd_typedef, simd_ovars, simd_ovar_inits, simd_ivecs, simd_ivec_inits, simd_iarrs,
             simd_iarr_inits, simd_uvar_inits,
             simd_stores) = self.__generateSimdCode(ovecs, ivecs, iarrs)

        # generate codes for the unrolled computation codes
        iarefs = [['%s[%s]' % (a,i) for i in range(0, ainfo.in_unroll_factor)] for a in iarrs]
        if self.vectorize:
            main_main_ops = self.__generateUnrolledOpCode(iarrs, ainfo.in_unroll_factor,
                                                          simd_ovars, simd_iarrs, simd_ivecs)
        else:
            main_main_ops = self.__generateUnrolledOpCode(iarrs, ainfo.in_unroll_factor,
                                                          ovecs, iarefs, ivecs)
        main_cleanup_ops = self.__generateUnrolledOpCode(iarrs, 1, ovecs, iarefs, ivecs[:1])
        if self.vectorize:
            cleanup_main_ops = self.__generateUnrolledOpCode(iarrs[:1], ainfo.in_unroll_factor,
                                                             simd_ovars[:1], simd_iarrs[:1],
                                                             simd_ivecs)
        else:
            cleanup_main_ops = self.__generateUnrolledOpCode(iarrs[:1], ainfo.in_unroll_factor,
                                                             ovecs[:1], iarefs[:1], ivecs)
        cleanup_cleanup_ops = self.__generateUnrolledOpCode(iarrs[:1], 1,
                                                            ovecs[:1], iarefs[:1], ivecs[:1])

        # generate code for the main unrolled loop
        code = ''
        if self.vectorize:
            code += simd_typedef + '\n'
        code += ' '.join(iarr_decls) + '\n'
        code += ' '.join(ovec_decls) + '\n'
        if self.vectorize:
            code += ivec_decls[0] + '\n'
        else:
            code += ' '.join(ivec_decls) + '\n'
        if self.include_outer_loop:
            code += 'register int %s=0;\n' % ainfo.out_loop_var
            code += 'while (%s<=%s-%s) {\n' % (ainfo.out_loop_var, ainfo.num_rows,
                                               (1 + ainfo.out_unroll_factor - 1))
        code += '  ' + ' '.join(iarr_inits) + '\n'
        if self.vectorize:
            code += '  ' + ' '.join(simd_ovar_inits) + '\n'
        else:
            code += '  ' + ' '.join(ovec_inits) + '\n'
        code += '  ' + 'register int %s=0;\n' % ainfo.in_loop_var
        code += '  ' + 'while (%s<=%s-%s) {\n' % (ainfo.in_loop_var, ainfo.num_cols,
                                                  (1 + ainfo.in_unroll_factor - 1))
        if self.vectorize:
            code += '    ' + '\n    '.join(simd_ivec_inits) + '\n'
        else:
            code += '    ' + ' '.join(ivec_inits) + '\n'
        code += '    ' + cind_unroll_inc + '\n'
        if self.vectorize:
            code += '    ' + '\n    '.join(simd_iarr_inits) + '\n'
        code += '    ' + '\n    '.join(main_main_ops) + '\n'
        code += '    ' + '%s+=%s;\n' % (ainfo.in_loop_var, ainfo.in_unroll_factor)
        code += '  ' + '} \n'
        if self.vectorize:
            code += '  ' + ' '.join(simd_uvar_inits) + '\n'
            code += '  ' + ' '.join(simd_stores) + '\n'
        if ainfo.in_unroll_factor > 1:
            code += '  ' + 'while (%s<=%s-1) {\n' % (ainfo.in_loop_var, ainfo.num_cols)
            code += '    ' + ivec_inits[0] + '\n'
            code += '    ' + cind_cleanup_inc + '\n'
            code += '    ' + '\n    '.join(main_cleanup_ops) + '\n'
            code += '    ' + '%s++;\n' % ainfo.in_loop_var
            code += '  ' + '} \n'
        code += '  ' + ' '.join(ovec_stores) + '\n'
        code += '  ' + ovec_unroll_inc + '\n'
        code += '  ' + iarr_unroll_inc + '\n'
        code += '  ' + cind_inc + '\n'
        if self.include_outer_loop:
            code += '  ' + '%s+=%s;\n' % (ainfo.out_loop_var, ainfo.out_unroll_factor)
            code += '} \n'
        
        # generate code for the clean-up loop
        if ainfo.out_unroll_factor > 1 and self.include_cleanup_code:
            code += 'while (%s<=%s-1) {\n' % (ainfo.out_loop_var, ainfo.num_rows)
            code += '  ' + iarr_inits[0] + '\n'
            if self.vectorize:
                code += '  ' + simd_ovar_inits[0] + '\n'
            else:
                code += '  ' + ovec_inits[0] + '\n'
            code += '  ' + 'register int %s=0;\n' % ainfo.in_loop_var
            code += '  ' + 'while (%s<=%s-%s) {\n' % (ainfo.in_loop_var, ainfo.num_cols,
                                                      (1 + ainfo.in_unroll_factor - 1))
            if self.vectorize:
                code += '    ' + '\n    '.join(simd_ivec_inits) + '\n'
            else:
                code += '    ' + ' '.join(ivec_inits) + '\n'
            code += '    ' + cind_unroll_inc + '\n'
            if self.vectorize:
                code += '    ' + simd_iarr_inits[0] + '\n'
            code += '    ' + '\n    '.join(cleanup_main_ops) + '\n'
            code += '    ' + '%s+=%s;\n' % (ainfo.in_loop_var, ainfo.in_unroll_factor)
            code += '  ' + '} \n'
            if self.vectorize:
                code += '  ' + simd_uvar_inits[0] + '\n'
                code += '  ' + simd_stores[0] + '\n'
            if ainfo.in_unroll_factor > 1:
                code += '  ' + 'while (%s<=%s-1) {\n' % (ainfo.in_loop_var, ainfo.num_cols)
                code += '    ' + ivec_inits[0] + '\n'
                code += '    ' + cind_cleanup_inc + '\n'
                code += '    ' + '\n    '.join(cleanup_cleanup_ops) + '\n'
                code += '    ' + '%s++;\n' % ainfo.in_loop_var
                code += '  ' + '} \n'
            code += '  ' + ovec_stores[0] + '\n'
            code += '  ' + ovec_cleanup_inc + '\n'
            code += '  ' + iarr_cleanup_inc + '\n'
            code += '  ' + '%s++;\n' % ainfo.out_loop_var
            code += '}'

        # add a new scope and indentation
        code = '\n{ \n' + '  ' + re.sub('\n', '\n  ', code) + '\n} \n'
        
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
    
