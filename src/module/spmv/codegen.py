#
# The code generator for the SpMV transformation module
#

import re, sys
import arg_info

#-------------------------------------------

class CodeGen:
    '''The code generator for the SpMV transformation module'''

    def __init__(self, ainfo):
        '''To instantiate a code generator instance'''
        
        self.ainfo = ainfo

        # check semantics
        self.__semantCheck()
        
    #------------------------------------------------------

    def __semantCheck(self):
        '''Check the semantic correctness'''

        # get the argument information
        ainfo = self.ainfo

        # semantic for vectorization
        if (ainfo.simd != arg_info.ArgInfo.SIMD_NONE and ainfo.in_unroll_factor > 1 and
            ainfo.in_unroll_factor % 2 != 0):
            print ('error:SpMV: simdization requires the inner loop unroll factor to be ' +
                   'divisible by two')
            sys.exit(1)

    #------------------------------------------------------

    def __getIArrs(self):
        '''Get the input arrays'''

        ainfo = self.ainfo
        if ainfo.out_unroll_factor == 1:
            iarrs = [ainfo.in_matrix]
            iarr_inits = []
        else:
            iarrs = ['%s%s' % (ainfo.in_matrix, i) for i in range(0, ainfo.out_unroll_factor)]
            iarr_inits = [('%s=%s' % (a, ainfo.in_matrix)) if i==0 else
                          ('%s=%s+clength' % (a, iarrs[i-1]))
                          for i,a in enumerate(iarrs)]
        return (iarrs, iarr_inits)
        
    def __getOVecs(self):
        '''Get the output vectors'''

        ainfo = self.ainfo
        ovecs = ['%s%s' % (ainfo.out_vector, i) for i in range(0, ainfo.out_unroll_factor)]
        ovec_inits = ['%s=%s' % (v, ainfo.init_val) for v in ovecs]
        ovec_stores = ['%s[%s]=%s' % (ainfo.out_vector,i,v) for i,v in enumerate(ovecs)]
        return (ovecs, ovec_inits, ovec_stores)
        
    def __getIVecs(self):
        '''Get the input vectors'''

        ainfo = self.ainfo
        ivecs = ['%s%s' % (ainfo.in_vector, i) for i in range(0, ainfo.in_unroll_factor)]
        ivec_refs = ['%s[%s[%s]]' % (ainfo.in_vector,ainfo.col_inds,i)
                     for i in range(0,ainfo.in_unroll_factor)]
        return (ivecs, ivec_refs)

    #------------------------------------------------------

    def __getSimdIArrs(self, iarrs):
        '''Get the simd input arrays'''

        ainfo = self.ainfo
        viarrs = ['%sv' % a for a in iarrs]
        viarr_inits = ['%s=(v2df *)%s' % (r,a) for r,a in zip(viarrs,iarrs)]
        return (viarrs, viarr_inits)

    def __getSimdOVecs(self, ovecs):
        '''Get the simd output vectors'''

        ainfo = self.ainfo
        vovecs = ['%sv' % v for v in ovecs]
        vovec_inits = ['%s={%s,%s}' % (r,ainfo.init_val,ainfo.init_val) for r in vovecs]
        vtovecs = ['%su' % v for v in ovecs]
        vtovec_decls = ['*%s=(%s *)&%s' % (v,ainfo.data_type,r) for v,r in zip(vtovecs,vovecs)]
        vovec_stores = ['%s=%s[0]+%s[1]' % (r,t,t) for r,t in zip(ovecs,vtovecs)]
        return (vovecs, vovec_inits, vovec_stores, vtovec_decls)

    #------------------------------------------------------

    def __generateUnrolledBody(self, ovecs, iarrs, ivecs, ivec_refs):
        '''Generate a code for the unrolled loop body'''

        ainfo = self.ainfo
        code = ''
        used_ivecs = ivec_refs
        if len(ovecs) > 1:
            code += '      %s %s;\n' % (ainfo.data_type,
                                        ','.join(['%s=%s' % (v,r) for v,r in zip(ivecs, ivec_refs)]))
            used_ivecs = ivecs
        for ov,ia in zip(ovecs, iarrs):
            code += '      %s += ' % ov
            for i,iv in enumerate(used_ivecs):
                if i: code += ' + '
                code += '%s[%s]*%s' % (ia,i,iv)
            code += ';\n'
        return code

    #------------------------------------------------------

    def __generateUnrolledSimdBody(self, vovecs, viarrs, ivec_refs):
        '''Generate a simdized code for the unrolled loop body'''

        ainfo = self.ainfo
        vivecs = ['%s%sv' % (ainfo.in_vector, i) for i in range(0, ainfo.in_unroll_factor/2)]
        code = ''
        code += '      v2df %s;\n' % ','.join('%s={%s,%s}' % (vivecs[i], ivec_refs[2*i],
                                                              ivec_refs[2*i+1])
                                              for i in range(0, ainfo.in_unroll_factor/2))
        for vov,via in zip(vovecs, viarrs):
            code += '      %s += ' % vov
            for i,viv in enumerate(vivecs):
                if i: code += ' + '
                code += '%s[%s]*%s' % (via,i,viv)
            code += ';\n'
        return code

    #------------------------------------------------------

    def __generateInodeCode(self):
        '''Generate an inode optimized code that is sequential'''

        # get the argument information
        ainfo = self.ainfo

        # get other important information
        iarrs, iarr_inits = self.__getIArrs()
        ovecs, ovec_inits, ovec_stores = self.__getOVecs()
        ivecs, ivec_refs = self.__getIVecs()

        # get the unrolled loop bodies
        main_main_ucode = self.__generateUnrolledBody(ovecs, iarrs, ivecs, ivec_refs)
        main_cleanup_ucode = self.__generateUnrolledBody(ovecs, iarrs, ivecs[:1], ivec_refs[:1])
        cleanup_main_ucode = self.__generateUnrolledBody(ovecs[:1], [ainfo.in_matrix],
                                                         ivecs, ivec_refs)
        cleanup_cleanup_ucode = self.__generateUnrolledBody(ovecs[:1], [ainfo.in_matrix],
                                                            ivecs[:1], ivec_refs[:1])

        # generate the optimized code
        code = ''
        code += 'register int rlength, clength;\n'
        code += 'register int n=%s;\n' % ainfo.total_inodes
        code += 'while (n--) {\n'
        code += '  rlength=%s[0]; %s++;\n' % (ainfo.inode_row_sizes, ainfo.inode_row_sizes)
        code += '  clength=%s[1]-%s[0]; %s+=rlength;\n' % (ainfo.row_inds, ainfo.row_inds,
                                                           ainfo.row_inds)
        code += '  register int i=0;\n'
        code += '  while (i<=rlength-%s) {\n' % ainfo.out_unroll_factor
        if ainfo.out_unroll_factor > 1:
            code += '    %s %s;\n' % (ainfo.data_type, ','.join(map(lambda x:'*'+x, iarr_inits)))
        code += '    %s %s;\n' % (ainfo.data_type, ','.join(ovec_inits))
        code += '    register int j=0;\n'
        code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
        code += main_main_ucode
        code += '      %s;\n' % '; '.join(['%s+=%s' % (a, ainfo.in_unroll_factor) for a in iarrs])
        code += '      %s+=%s;\n' % (ainfo.col_inds, ainfo.in_unroll_factor)
        code += '      j+=%s;\n' % ainfo.in_unroll_factor
        code += '    }\n'
        if ainfo.in_unroll_factor > 1:
            code += '    while (j<=clength-1) {\n'
            code += main_cleanup_ucode
            code += '      %s;\n' % '; '.join(['%s++' % a for a in iarrs])
            code += '      %s++;\n' % ainfo.col_inds
            code += '      j++;\n'
            code += '    }\n'
        code += '    %s;\n' % '; '.join(ovec_stores)
        code += '    y+=%s;\n' % ainfo.out_unroll_factor
        if ainfo.out_unroll_factor > 1:
            code += '    %s=%s;\n' % (ainfo.in_matrix, iarrs[-1])
            if ainfo.out_unroll_factor == 2:
                code += '    %s+=clength;\n' % ainfo.col_inds
            else:
                code += '    %s+=%s*clength;\n' % (ainfo.col_inds, ainfo.out_unroll_factor-1)
        code += '    i+=%s;\n' % ainfo.out_unroll_factor
        code += '  }\n'
        if ainfo.out_unroll_factor > 1:
            code += '  while (i<=rlength-1) {\n'
            code += '    %s %s;\n' % (ainfo.data_type, ovec_inits[0])
            code += '    register int j=0;\n'
            code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
            code += cleanup_main_ucode
            code += '      %s+=%s;\n' % (ainfo.in_matrix, ainfo.in_unroll_factor)
            code += '      %s+=%s;\n' % (ainfo.col_inds, ainfo.in_unroll_factor)
            code += '      j+=%s;\n' % ainfo.in_unroll_factor
            code += '    }\n'
            if ainfo.in_unroll_factor > 1:
                code += '    while (j<=clength-1) {\n'
                code += cleanup_cleanup_ucode
                code += '      %s++;\n' % ainfo.in_matrix
                code += '      %s++;\n' % ainfo.col_inds
                code += '      j++;\n'
                code += '    }\n'
            code += '    %s;\n' % ovec_stores[0]
            code += '    y++;\n'
            code += '    i++;\n'
            code += '  }\n'
        code += '}\n'

        # to enclose with brackets and to correct indentation
        code = '\n{\n' + re.sub('\n', '\n  ', '\n' + code) + '\n}\n'

        # return the generated code
        return code
        
    #------------------------------------------------------

    def __generateSimdInodeCode(self):
        '''Generate an inode optimized code that is sequential and simdized'''

        # get the argument information
        ainfo = self.ainfo

        # get other important information
        iarrs, iarr_inits = self.__getIArrs()
        ovecs, ovec_inits, ovec_stores = self.__getOVecs()
        ivecs, ivec_refs = self.__getIVecs()
        viarrs, viarr_inits = self.__getSimdIArrs(iarrs)
        vovecs, vovec_inits, vovec_stores, vtovec_decls = self.__getSimdOVecs(ovecs)

        # get the unrolled loop bodies
        main_main_ucode = self.__generateUnrolledSimdBody(vovecs, viarrs, ivec_refs)
        main_cleanup_ucode = self.__generateUnrolledBody(ovecs, iarrs, ivecs[:1], ivec_refs[:1])
        cleanup_main_ucode = self.__generateUnrolledSimdBody(vovecs[:1], ['%sv' % ainfo.in_matrix],
                                                             ivec_refs)
        cleanup_cleanup_ucode = self.__generateUnrolledBody(ovecs[:1], [ainfo.in_matrix],
                                                            ivecs[:1], ivec_refs[:1])

        # generate the optimized code
        code = ''
        code += 'typedef double v2df __attribute__ ((vector_size(16)));\n'
        code += 'register int rlength, clength;\n'
        code += 'register int n=%s;\n' % ainfo.total_inodes
        code += 'while (n--) {\n'
        code += '  rlength=%s[0]; %s++;\n' % (ainfo.inode_row_sizes, ainfo.inode_row_sizes)
        code += '  clength=%s[1]-%s[0]; %s+=rlength;\n' % (ainfo.row_inds, ainfo.row_inds,
                                                           ainfo.row_inds)
        code += '  register int i=0;\n'
        code += '  while (i<=rlength-%s) {\n' % ainfo.out_unroll_factor
        if ainfo.out_unroll_factor > 1:
            code += '    %s %s;\n' % (ainfo.data_type, ','.join(map(lambda x:'*'+x, iarr_inits)))
        code += '    v2df %s;\n' % ','.join([('*'+i) for i in viarr_inits])
        code += '    v2df %s;\n' % ','.join(vovec_inits)
        code += '    register int j=0;\n'
        code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
        code += main_main_ucode
        code += '      %s;\n' % '; '.join(['%s+=%s' % (a,ainfo.in_unroll_factor/2) for a in viarrs])
        code += '      %s+=%s;\n' % (ainfo.col_inds, ainfo.in_unroll_factor)
        code += '      j+=%s;\n' % ainfo.in_unroll_factor
        code += '    }\n'
        code += '    %s;\n' % '; '.join(['%s+=j' % a for a in iarrs])
        code += '    %s %s;\n' % (ainfo.data_type, ','.join(vtovec_decls))
        code += '    %s %s;\n' % (ainfo.data_type, ','.join(vovec_stores))
        if ainfo.in_unroll_factor > 1:
            code += '    while (j<=clength-1) {\n'
            code += main_cleanup_ucode
            code += '      %s;\n' % '; '.join(['%s++' % a for a in iarrs])
            code += '      %s++;\n' % ainfo.col_inds
            code += '      j++;\n'
            code += '    }\n'
        code += '    %s;\n' % '; '.join(ovec_stores)
        code += '    y+=%s;\n' % ainfo.out_unroll_factor
        if ainfo.out_unroll_factor > 1:
            code += '    %s=%s;\n' % (ainfo.in_matrix, iarrs[-1])
            if ainfo.out_unroll_factor == 2:
                code += '    %s+=clength;\n' % ainfo.col_inds
            else:
                code += '    %s+=%s*clength;\n' % (ainfo.col_inds, ainfo.out_unroll_factor-1)
        code += '    i+=%s;\n' % ainfo.out_unroll_factor
        code += '  }\n'
        if ainfo.out_unroll_factor > 1:
            code += '  while (i<=rlength-1) {\n'
            code += '    v2df *%sv=(v2df *)%s;\n' % (ainfo.in_matrix, ainfo.in_matrix)
            code += '    v2df %s;\n' % vovec_inits[0]
            code += '    register int j=0;\n'
            code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
            code += cleanup_main_ucode
            code += '      %sv+=%s;\n' % (ainfo.in_matrix, ainfo.in_unroll_factor/2)
            code += '      %s+=%s;\n' % (ainfo.col_inds, ainfo.in_unroll_factor)
            code += '      j+=%s;\n' % ainfo.in_unroll_factor
            code += '    }\n'
            code += '    %s+=j;\n' % ainfo.in_matrix
            code += '    %s %s;\n' % (ainfo.data_type, vtovec_decls[0])
            code += '    %s %s;\n' % (ainfo.data_type, vovec_stores[0])
            if ainfo.in_unroll_factor > 1:
                code += '    while (j<=clength-1) {\n'
                code += cleanup_cleanup_ucode
                code += '      %s++;\n' % ainfo.in_matrix
                code += '      %s++;\n' % ainfo.col_inds
                code += '      j++;\n'
                code += '    }\n'
            code += '    %s;\n' % ovec_stores[0]
            code += '    y++;\n'
            code += '    i++;\n'
            code += '  }\n'
        code += '}\n'

        # to enclose with brackets and to correct indentation
        code = '\n{\n' + re.sub('\n', '\n  ', '\n' + code) + '\n}\n'

        # return the generated code
        return code
        
    #------------------------------------------------------

    def __generateParCode(self, parallelize=False):
        '''Generate an optimized code that is parallel'''
        return self.__generateCode(True)

    def __generateCode(self, parallelize=False):
        '''Generate an optimized code that is sequential'''

        # get the argument information
        ainfo = self.ainfo

        # generate the optimized code
        code = ''
        code += 'register int i;\n'
        if parallelize:
            if ainfo.out_unroll_factor==1:
                code += 'register int lbound=%s;\n' % ainfo.total_rows
            else:
                code += 'register int lbound=%s-(%s%%%s);\n' % (ainfo.total_rows, ainfo.total_rows,
                                                                ainfo.out_unroll_factor)
            code += 'omp_set_num_threads(%s);\n' % ainfo.num_threads
            code += ('#pragma omp parallel for shared(%s,%s,%s,%s,%s,%s,lbound) private(i)\n' %
                     (ainfo.out_vector, ainfo.in_vector, ainfo.in_matrix, ainfo.row_inds,
                      ainfo.col_inds, ainfo.total_rows))
            code += 'for (i=0; i<=lbound-%s; i+=%s) {\n' % (ainfo.out_unroll_factor,
                                                            ainfo.out_unroll_factor)
        else:
            code += 'for (i=0; i<=%s-%s; i+=%s) {\n' % (ainfo.total_rows, ainfo.out_unroll_factor,
                                                        ainfo.out_unroll_factor)
        code += '  register int %s;\n' % ','.join(['lb%s=%s[i%s]' % (i, ainfo.row_inds,
                                                                     '+%s'%i if i else '')
                                                   for i in range(0, ainfo.out_unroll_factor+1)])
        for i in range(0, ainfo.out_unroll_factor):
            code += '  %s%s0=%s;\n' % ('' if i else '%s '%ainfo.data_type, ainfo.out_vector,
                                       ainfo.init_val)
            code += '  %sj=lb%s;\n' % ('' if i else 'register int ', i)
            code += '  while (j<=lb%s-%s) {\n' % (i+1, ainfo.in_unroll_factor)
            code += '    %s0 += ' % ainfo.out_vector
            for j in range(0, ainfo.in_unroll_factor):
                if j: code += ' + '
                code += '%s[j%s]*%s[%s[j%s]]' % (ainfo.in_matrix, '+%s'%j if j else '',
                                                 ainfo.in_vector, ainfo.col_inds,
                                                 '+%s'%j if j else '')
            code += ';\n'
            code += '    j+=%s;\n' % ainfo.in_unroll_factor
            code += '  }\n'
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=lb%s-1) {\n' % (i+1)
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j++;\n'
                code += '  }\n'
            code += '  %s[i%s]=%s0;\n' % (ainfo.out_vector, '+%s'%i if i else '', ainfo.out_vector)
        code += '} \n'
        if ainfo.out_unroll_factor > 1:
            if parallelize:
                code += 'i=lbound;\n'
            code += 'while (i<=%s-1) {\n' % ainfo.total_rows
            code += '  %s %s0=%s;\n' % (ainfo.data_type, ainfo.out_vector, ainfo.init_val)
            code += '  register int j=%s[i], ub=%s[i+1];\n' % (ainfo.row_inds, ainfo.row_inds)
            code += '  while (j<=ub-%s) {\n' % ainfo.in_unroll_factor
            code += '    %s0 += ' % ainfo.out_vector
            for j in range(0, ainfo.in_unroll_factor):
                if j: code += ' + '
                code += '%s[j%s]*%s[%s[j%s]]' % (ainfo.in_matrix, '+%s'%j if j else '',
                                                 ainfo.in_vector, ainfo.col_inds,
                                                 '+%s'%j if j else '')
            code += ';\n'
            code += '    j+=%s;\n' % ainfo.in_unroll_factor
            code += '  }\n'
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=ub-1) {\n'
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j++;\n'
                code += '  }\n'   
            code += '  %s[i]=%s0;\n' % (ainfo.out_vector, ainfo.out_vector)
            code += '  i++;\n'
            code += '}\n'

        # to enclose with brackets and to correct indentation
        code = '\n{\n' + re.sub('\n', '\n  ', '\n' + code) + '\n}\n'

        # return the generated code
        return code

    #------------------------------------------------------

    def __generateParSimdCode(self):
        '''Generate an optimized code that is parallel and simdized'''
        return self.__generateSimdCode(True)

    def __generateSimdCode(self, parallelize=False):
        '''Generate an optimized code that is sequential and simdized'''

        # get the argument information
        ainfo = self.ainfo

        # generate the optimized code
        code = ''
        code += 'typedef double v2df __attribute__ ((vector_size(16)));\n'
        code += 'register int i;\n'
        if parallelize:
            if ainfo.out_unroll_factor==1:
                code += 'register int lbound=%s;\n' % ainfo.total_rows
            else:
                code += 'register int lbound=%s-(%s%%%s);\n' % (ainfo.total_rows, ainfo.total_rows,
                                                                ainfo.out_unroll_factor)
            code += 'omp_set_num_threads(%s);\n' % ainfo.num_threads
            code += ('#pragma omp parallel for shared(%s,%s,%s,%s,%s,%s,lbound) private(i)\n' %
                     (ainfo.out_vector, ainfo.in_vector, ainfo.in_matrix, ainfo.row_inds,
                      ainfo.col_inds, ainfo.total_rows))
            code += 'for (i=0; i<=lbound-%s; i+=%s) {\n' % (ainfo.out_unroll_factor,
                                                            ainfo.out_unroll_factor)
        else:
            code += 'for (i=0; i<=%s-%s; i+=%s) {\n' % (ainfo.total_rows, ainfo.out_unroll_factor,
                                                        ainfo.out_unroll_factor)
        code += '  register int %s;\n' % ','.join(['lb%s=%s[i%s]' % (i, ainfo.row_inds,
                                                                     '+%s'%i if i else '')
                                                   for i in range(0, ainfo.out_unroll_factor+1)])
        for i in range(0, ainfo.out_unroll_factor):
            code += '  %s%s0v=(v2df){%s,%s};\n' % ('' if i else 'v2df ', ainfo.out_vector,
                                                   ainfo.init_val, ainfo.init_val)
            code += '  %sj=lb%s;\n' % ('' if i else 'register int ', i)
            code += '  %s%s0v=(v2df *)(%s+j);\n' % ('' if i else 'v2df *',
                                                    ainfo.in_matrix, ainfo.in_matrix) 
            code += '  while (j<=lb%s-%s) {\n' % (i+1, ainfo.in_unroll_factor)
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '    v2df '
                if j: code += ','
                code += ('%s%sv={%s[%s[j%s]],%s[%s[j%s]]}' %
                         (ainfo.in_vector, j,
                          ainfo.in_vector, ainfo.col_inds, '+%s'%(2*j) if (2*j) else '',
                          ainfo.in_vector, ainfo.col_inds, '+%s'%(2*j+1) if (2*j+1) else '',))
            code += ';\n'
            code += '    %s0v += ' % ainfo.out_vector
            for j in range(0, ainfo.in_unroll_factor/2):
                if j: code += ' + '
                code += '%s0v[%s]*%s%sv' % (ainfo.in_matrix, j, ainfo.in_vector, j)
            code += ';\n'
            code += '    j+=%s; %s0v+=%s;\n' % (ainfo.in_unroll_factor, ainfo.in_matrix,
                                                ainfo.in_unroll_factor/2)
            code += '  }\n'
            code += '  %s%s0u=(%s *)&%s0v;\n' % ('' if i else '%s *'%ainfo.data_type,
                                                 ainfo.out_vector, ainfo.data_type, ainfo.out_vector)
            code += '  %s%s0=%s0u[0]+%s0u[1];\n' % ('' if i else '%s '%ainfo.data_type,
                                                    ainfo.out_vector, ainfo.out_vector,
                                                    ainfo.out_vector)
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=lb%s-1) {\n' % (i+1)
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j++;\n'
                code += '  }\n'
            code += '  %s[i%s]=%s0;\n' % (ainfo.out_vector, '+%s'%i if i else '', ainfo.out_vector)
        code += '} \n'
        if ainfo.out_unroll_factor > 1:
            if parallelize:
                code += 'i=lbound;\n'
            code += 'while (i<=%s-1) {\n' % ainfo.total_rows
            code += '  v2df %s0v=(v2df){%s,%s};\n' % (ainfo.out_vector, ainfo.init_val,
                                                      ainfo.init_val)
            code += '  register int j=%s[i], ub=%s[i+1];\n' % (ainfo.row_inds, ainfo.row_inds)
            code += '  v2df *%s0v=(v2df *)(%s+j);\n' % (ainfo.in_matrix, ainfo.in_matrix)
            code += '  while (j<=ub-%s) {\n' % ainfo.in_unroll_factor
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '    v2df '
                if j: code += ','
                code += ('%s%sv={%s[%s[j%s]],%s[%s[j%s]]}' %
                         (ainfo.in_vector, j,
                          ainfo.in_vector, ainfo.col_inds, '+%s'%(2*j) if (2*j) else '',
                          ainfo.in_vector, ainfo.col_inds, '+%s'%(2*j+1) if (2*j+1) else '',))
            code += ';\n'
            code += '    %s0v += ' % ainfo.out_vector
            for j in range(0, ainfo.in_unroll_factor/2):
                if j: code += ' + '
                code += '%s0v[%s]*%s%sv' % (ainfo.in_matrix, j, ainfo.in_vector, j)
            code += ';\n'
            code += '    j+=%s; %s0v+=%s;\n' % (ainfo.in_unroll_factor, ainfo.in_matrix,
                                                ainfo.in_unroll_factor/2)
            code += '  }\n'
            code += '  %s *%s0u=(%s *)&%s0v;\n' % (ainfo.data_type, ainfo.out_vector,
                                                   ainfo.data_type, ainfo.out_vector)
            code += '  %s %s0=%s0u[0]+%s0u[1];\n' % (ainfo.data_type, ainfo.out_vector,
                                                     ainfo.out_vector, ainfo.out_vector)
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=ub-1) {\n'
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j++;\n'
                code += '  }\n'   
            code += '  %s[i]=%s0;\n' % (ainfo.out_vector, ainfo.out_vector)
            code += '  i++;\n'
            code += '}\n'

        # to enclose with brackets and to correct indentation
        code = '\n{\n' + re.sub('\n', '\n  ', '\n' + code) + '\n}\n'

        # return the generated code
        return code

    #------------------------------------------------------

    def generate(self):
        '''To generate optimized SpMV code'''

        # the argument information
        ainfo = self.ainfo

        # generate the optimized code
        if ainfo.block_structure == arg_info.ArgInfo.BSTRUC_NONE:
            if ainfo.num_threads == 1:
                if ainfo.simd == arg_info.ArgInfo.SIMD_NONE:
                    code = self.__generateCode()
                else:
                    code = self.__generateSimdCode()
            else:
                if ainfo.simd == arg_info.ArgInfo.SIMD_NONE:
                    code = self.__generateParCode()
                else:
                    code = self.__generateParSimdCode()
        elif ainfo.block_structure == arg_info.ArgInfo.BSTRUC_INODE:
            if ainfo.num_threads == 1:
                if ainfo.simd == arg_info.ArgInfo.SIMD_NONE:
                    code = self.__generateInodeCode()
                else:
                    code = self.__generateSimdInodeCode()
            else:
                if ainfo.simd == arg_info.ArgInfo.SIMD_NONE:
                    code = self.__generateParCode()
                else:
                    code = self.__generateParSimdCode()
        else:
            print 'error:SpMV: unsupported matrix block structure'
            sys.exit(1)

        # return the optimized code
        return code

