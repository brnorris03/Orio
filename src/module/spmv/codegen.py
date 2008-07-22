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

    def __generateParInodeCode(self):
        '''Generate an inode optimized code that is parallel'''
        return self.__generateInodeCode(True)

    def __generateInodeCode(self, parallelize=False):
        '''Generate an inode optimized code that is sequential'''

        # get the argument information
        ainfo = self.ainfo

        # start generating the optimized code
        code = ''

        # the inode loop
        if parallelize:
            iarr = '%sc' % ainfo.in_matrix
            iarrs = [iarr] + \
                    ['%s%sc' % (ainfo.in_matrix, i) for i in range(1, ainfo.out_unroll_factor)]
            iarr_inits = ['%s=%s+clength' % (a, iarrs[i]) for i,a in enumerate(iarrs[1:])]
            ovec = '%sc' % ainfo.out_vector
            ovecs = ['%s%sc' % (ainfo.out_vector, i) for i in range(0, ainfo.out_unroll_factor)]
            ovec_inits = ['%s=%s' % (v, ainfo.init_val) for v in ovecs]
            ovec_stores = ['%sc[%s]=%s' % (ainfo.out_vector,i,v) for i,v in enumerate(ovecs)]
            col_inds = '%sc' % ainfo.col_inds

            code += 'register int n;\n'
            code += 'omp_set_num_threads(%s);\n' % ainfo.num_threads
            code += ('#pragma omp parallel for shared(%s,%s,%s,%s,%s,%s,%s) private(n)\n' %
                     (ainfo.out_vector, ainfo.in_vector, ainfo.in_matrix, ainfo.row_inds,
                      ainfo.col_inds, ainfo.total_inodes, ainfo.inode_rows))
            code += 'for (n=0; n<=%s-1; n+=1) {\n' % ainfo.total_inodes
            code += '  int start_row=%s[n];\n' % ainfo.inode_rows
            code += '  register int rlength=%s[n+1]-start_row;\n' % ainfo.inode_rows
            code += '  int first_col=%s[start_row];\n' % ainfo.row_inds
            code += '  register int clength=%s[start_row+1]-first_col;\n' % ainfo.row_inds
            code += '  %s *%s=&%s[start_row];\n' % (ainfo.data_type, ovec, ainfo.out_vector)
            code += '  int *%s=&%s[first_col];\n' % (col_inds, ainfo.col_inds)
            code += '  %s *%s=&%s[first_col];\n' % (ainfo.data_type, iarr, ainfo.in_matrix)

        else:
            iarr = ainfo.in_matrix
            iarrs = [iarr] + \
                    ['%s%s' % (ainfo.in_matrix, i) for i in range(1, ainfo.out_unroll_factor)]
            iarr_inits = ['%s=%s+clength' % (a, iarrs[i]) for i,a in enumerate(iarrs[1:])]
            ovec = ainfo.out_vector
            ovecs = ['%s%s' % (ainfo.out_vector, i) for i in range(0, ainfo.out_unroll_factor)]
            ovec_inits = ['%s=%s' % (v, ainfo.init_val) for v in ovecs]
            ovec_stores = ['%s[%s]=%s' % (ainfo.out_vector,i,v) for i,v in enumerate(ovecs)]
            col_inds = ainfo.col_inds

            code += 'register int n=%s;\n' % ainfo.total_inodes
            code += 'while (n--) {\n'
            code += '  register int rlength=%s[0]; %s+=1;\n' % (ainfo.inode_sizes, ainfo.inode_sizes)
            code += '  register int clength=%s[1]-%s[0]; %s+=rlength;\n' % (ainfo.row_inds,
                                                                            ainfo.row_inds,
                                                                            ainfo.row_inds)
            
        ivecs = ['%s%s' % (ainfo.in_vector, i) for i in range(0, ainfo.in_unroll_factor)]
        ivec_refs = ['%s[%s[%s]]' % (ainfo.in_vector, col_inds, i)
                     for i in range(0, ainfo.in_unroll_factor)]

        # the outer main loop
        code += '  register int i=0;\n'
        code += '  while (i<=rlength-%s) {\n' % ainfo.out_unroll_factor
        if len(iarr_inits) > 0:
            code += '    %s %s;\n' % (ainfo.data_type, ','.join(map(lambda x:'*'+x, iarr_inits)))
        code += '    %s %s;\n' % (ainfo.data_type, ','.join(ovec_inits))

        # the inner main-main loop
        code += '    register int j=0;\n'
        code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
        used_ivecs = ivec_refs
        if ainfo.out_unroll_factor > 1:
            code += '      %s %s;\n' % (ainfo.data_type,
                                        ','.join(['%s=%s' % (v,r) for v,r in zip(ivecs, ivec_refs)]))
            used_ivecs = ivecs
        for ov,ia in zip(ovecs, iarrs):
            code += '      %s += ' % ov
            for i,iv in enumerate(used_ivecs):
                if i: code += ' + '
                code += '%s[%s]*%s' % (ia,i,iv)
            code += ';\n'
        code += '      %s;\n' % '; '.join(['%s+=%s' % (a, ainfo.in_unroll_factor) for a in iarrs])
        code += '      %s+=%s;\n' % (col_inds, ainfo.in_unroll_factor)
        code += '      j+=%s;\n' % ainfo.in_unroll_factor
        code += '    }\n'

        # the inner main-cleanup loop
        if ainfo.in_unroll_factor > 1:
            code += '    while (j<=clength-1) {\n'
            used_ivecs = ivec_refs[:1]
            if ainfo.out_unroll_factor > 1:
                code += '      %s %s=%s;\n' % (ainfo.data_type, ivecs[0], ivec_refs[0])
                used_ivecs = ivecs[:1]
            for ov,ia in zip(ovecs, iarrs):
                code += '      %s += ' % ov
                for i,iv in enumerate(used_ivecs):
                    if i: code += ' + '
                    code += '%s[%s]*%s' % (ia,i,iv)
                code += ';\n'
            code += '      %s;\n' % '; '.join(['%s+=1' % a for a in iarrs])
            code += '      %s+=1;\n' % col_inds
            code += '      j+=1;\n'
            code += '    }\n'

        # the epiloque of the outer main loop
        code += '    %s;\n' % '; '.join(ovec_stores)
        code += '    %s+=%s;\n' % (ovec, ainfo.out_unroll_factor)
        if ainfo.out_unroll_factor > 1:
            code += '    %s=%s;\n' % (iarr, iarrs[-1])
            if ainfo.out_unroll_factor == 2:
                code += '    %s+=clength;\n' % col_inds
            else:
                code += '    %s+=%s*clength;\n' % (col_inds, ainfo.out_unroll_factor-1)
        code += '    i+=%s;\n' % ainfo.out_unroll_factor
        code += '  }\n'

        # the outer cleanup loop
        if ainfo.out_unroll_factor > 1:
            code += '  while (i<=rlength-1) {\n'
            code += '    %s %s;\n' % (ainfo.data_type, ovec_inits[0])

            # the inner cleanup-main loop
            code += '    register int j=0;\n'
            code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
            for ov,ia in zip(ovecs[:1], iarrs[:1]):
                code += '      %s += ' % ov
                for i,iv in enumerate(ivec_refs):
                    if i: code += ' + '
                    code += '%s[%s]*%s' % (ia,i,iv)
                code += ';\n'
            code += '      %s+=%s;\n' % (iarr, ainfo.in_unroll_factor)
            code += '      %s+=%s;\n' % (col_inds, ainfo.in_unroll_factor)
            code += '      j+=%s;\n' % ainfo.in_unroll_factor
            code += '    }\n'

            # the inner cleanup-cleanup loop
            if ainfo.in_unroll_factor > 1:
                code += '    while (j<=clength-1) {\n'
                for ov,ia in zip(ovecs[:1], iarrs[:1]):
                    code += '      %s += ' % ov
                    for i,iv in enumerate(ivec_refs[:1]):
                        if i: code += ' + '
                        code += '%s[%s]*%s' % (ia,i,iv)
                    code += ';\n'
                code += '      %s+=1;\n' % iarr
                code += '      %s+=1;\n' % col_inds
                code += '      j+=1;\n'
                code += '    }\n'

            # the epilogue of the outer cleanup loop
            code += '    %s;\n' % ovec_stores[0]
            code += '    %s+=1;\n' % ovec
            code += '    i+=1;\n'
            code += '  }\n'

        # close the inode loop
        code += '}\n'

        # to enclose with brackets and to correct indentation
        code = '\n{\n' + re.sub('\n', '\n  ', '\n' + code) + '\n}\n'

        # return the generated code
        return code
        
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

    def __generateSimdInodeCode(self):
        '''Generate an inode optimized code that is sequential and simdized'''

        # get the argument information
        ainfo = self.ainfo

        # generate the optimized code
        code = ''
        code += 'typedef double v2df __attribute__ ((vector_size(16)));\n'
        code += 'register int n=%s;\n' % ainfo.total_inodes
        code += 'while (n--) {\n'
        code += '  register int rlength=%s[0]; %s+=1;\n' % (ainfo.inode_sizes, ainfo.inode_sizes)
        code += '  register int clength=%s[1]-%s[0]; %s+=rlength;\n' % (ainfo.row_inds,
                                                                        ainfo.row_inds,
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
            code += '      %s;\n' % '; '.join(['%s+=1' % a for a in iarrs])
            code += '      %s+=1;\n' % ainfo.col_inds
            code += '      j+=1;\n'
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
                code += '      %s+=1;\n' % ainfo.in_matrix
                code += '      %s+=1;\n' % ainfo.col_inds
                code += '      j+=1;\n'
                code += '    }\n'
            code += '    %s;\n' % ovec_stores[0]
            code += '    y+=1;\n'
            code += '    i+=1;\n'
            code += '  }\n'
        code += '}\n'

        # to enclose with brackets and to correct indentation
        code = '\n{\n' + re.sub('\n', '\n  ', '\n' + code) + '\n}\n'

        # return the generated code
        return code
        
    #------------------------------------------------------

    def __generateParCode(self):
        '''Generate an optimized code that is parallel'''
        return self.__generateCode(True)

    def __generateCode(self, parallelize=False):
        '''Generate an optimized code that is sequential'''

        # get the argument information
        ainfo = self.ainfo

        # generate the optimized code
        code = ''

        # the outer main loop
        if parallelize:
            code += 'register int i;\n'
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
            code += 'register int i=0;\n'
            code += 'while (i<=%s-%s) {\n' % (ainfo.total_rows, ainfo.out_unroll_factor)

        code += '  register int %s;\n' % ','.join(['lb%s=%s[i%s]' % (i, ainfo.row_inds,
                                                                     '+%s'%i if i else '')
                                                   for i in range(0, ainfo.out_unroll_factor+1)])

        # the unrolled inner loop
        for i in range(0, ainfo.out_unroll_factor):
            code += '  %s%s0=%s;\n' % ('' if i else '%s ' % ainfo.data_type, ainfo.out_vector,
                                       ainfo.init_val)
            code += '  %sj=lb%s;\n' % ('' if i else 'register int ', i)

            # the inner main-main loop
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

            # the inner main-cleanup loop
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=lb%s-1) {\n' % (i+1)
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j+=1;\n'
                code += '  }\n'
            code += '  %s[i%s]=%s0;\n' % (ainfo.out_vector, '+%s'%i if i else '', ainfo.out_vector)

        # to close the outer main loop
        if not parallelize:
            code += '  i+=%s;\n' % ainfo.out_unroll_factor
        code += '} \n'

        # the outer cleanup loop
        if ainfo.out_unroll_factor > 1:
            if parallelize:
                code += 'i=lbound;\n'

            # the inner cleanup-main loop
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

            # the inner cleanup-cleanup loop
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=ub-1) {\n'
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j+=1;\n'
                code += '  }\n'   
            code += '  %s[i]=%s0;\n' % (ainfo.out_vector, ainfo.out_vector)
            code += '  i+=1;\n'
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

        # the outer main loop
        code += 'typedef double v2df __attribute__ ((vector_size(16)));\n'
        if parallelize:
            code += 'register int i;\n'
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
            code += 'register int i=0;\n'
            code += 'while (i<=%s-%s) {\n' % (ainfo.total_rows, ainfo.out_unroll_factor)
        code += '  register int %s;\n' % ','.join(['lb%s=%s[i%s]' % (i, ainfo.row_inds,
                                                                     '+%s'%i if i else '')
                                                   for i in range(0, ainfo.out_unroll_factor+1)])

        # the unrolled inner loop
        for i in range(0, ainfo.out_unroll_factor):
            code += '  %s%s0v=(v2df){%s,%s};\n' % ('' if i else 'v2df ', ainfo.out_vector,
                                                   ainfo.init_val, ainfo.init_val)

            # the inner main-main loop
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

            # the inner main-cleanup loop
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=lb%s-1) {\n' % (i+1)
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j+=1;\n'
                code += '  }\n'
            code += '  %s[i%s]=%s0;\n' % (ainfo.out_vector, '+%s'%i if i else '', ainfo.out_vector)

        # to close the outer main loop
        if not parallelize:
            code += '  i+=%s;\n' % ainfo.out_unroll_factor
        code += '} \n'

        # the outer cleanup loop
        if ainfo.out_unroll_factor > 1:
            if parallelize:
                code += 'i=lbound;\n'
            code += 'while (i<=%s-1) {\n' % ainfo.total_rows
            code += '  v2df %s0v=(v2df){%s,%s};\n' % (ainfo.out_vector, ainfo.init_val,
                                                      ainfo.init_val)
            code += '  register int j=%s[i], ub=%s[i+1];\n' % (ainfo.row_inds, ainfo.row_inds)
            code += '  v2df *%s0v=(v2df *)(%s+j);\n' % (ainfo.in_matrix, ainfo.in_matrix)

            # the inner cleanup-main loop
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

            # the inner cleanup-cleanup loop
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=ub-1) {\n'
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j+=1;\n'
                code += '  }\n'   
            code += '  %s[i]=%s0;\n' % (ainfo.out_vector, ainfo.out_vector)
            code += '  i+=1;\n'
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
                    code = self.__generateParInodeCode()
                else:
                    code = self.__generateParSimdInodeCode()
        else:
            print 'error:SpMV: unsupported matrix block structure'
            sys.exit(1)

        # return the optimized code
        return code

