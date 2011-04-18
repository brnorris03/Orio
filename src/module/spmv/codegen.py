#
# The code generator for the SpMV transformation module
#

import re, sys
import arg_info
from orio.main.util.globals import *

#-------------------------------------------

class CodeGen:
    '''The code generator for the SpMV transformation module.'''

    def __init__(self, ainfo):
        '''To instantiate a code generator instance'''

        self.ainfo = ainfo

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

        # the outer orio.main.loop
        code += '  register int i=0;\n'
        code += '  while (i<=rlength-%s) {\n' % ainfo.out_unroll_factor
        if len(iarr_inits) > 0:
            code += '    %s %s;\n' % (ainfo.data_type, ','.join(map(lambda x:'*'+x, iarr_inits)))
        code += '    %s %s;\n' % (ainfo.data_type, ','.join(ovec_inits))

        # the inner orio.main.main loop
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

        # the inner orio.main.cleanup loop
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

        # the epiloque of the outer orio.main.loop
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

            # the inner cleanup-orio.main.loop
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

    def __generateParGccSimdInodeCode(self):
        '''Generate an inode optimized code that is parallel and GCC simdized'''
        return self.__generateGccSimdInodeCode(True)

    def __generateGccSimdInodeCode(self,parallelize=False):
        '''Generate an inode optimized code that is sequential and GCC simdized'''
        
        # get the argument information
        ainfo = self.ainfo

        # start generating the optimized code
        code = ''

        # the inode loop
        code += 'typedef double v2df __attribute__ ((vector_size(16)));\n'
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
        viarr = '%sv' % iarr
        viarrs = ['%sv' % a for a in iarrs]
        viarr_inits = ['%s=(v2df *)%s' % (r,a) for r,a in zip(viarrs, iarrs)]
        vovecs = ['%sv' % v for v in ovecs]
        vovec_inits = ['%s={%s,%s}' % (r, ainfo.init_val, ainfo.init_val) for r in vovecs]
        vtovecs = ['%st' % v for v in ovecs]
        vtovec_decls = ['*%s=(%s *)&%s' % (v, ainfo.data_type, r) for v,r in zip(vtovecs, vovecs)]
        vovec_stores = ['%s=%s[0]+%s[1]' % (r, t, t) for r,t in zip(ovecs, vtovecs)]
        vivecs = ['%s%sv' % (ainfo.in_vector, i) for i in range(0, ainfo.in_unroll_factor/2)]
        
        # the outer orio.main.loop
        code += '  register int i=0;\n'
        code += '  while (i<=rlength-%s) {\n' % ainfo.out_unroll_factor
        if len(iarr_inits) > 0:
            code += '    %s %s;\n' % (ainfo.data_type, ','.join(map(lambda x:'*'+x, iarr_inits)))
        code += '    v2df %s;\n' % ','.join([('*'+i) for i in viarr_inits])
        code += '    v2df %s;\n' % ','.join(vovec_inits)

        # the inner orio.main.main loop
        code += '    register int j=0;\n'
        code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
        code += '      v2df %s;\n' % ','.join('%s={%s,%s}' % (vivecs[i], ivec_refs[2*i],
                                                              ivec_refs[2*i+1])
                                              for i in range(0, ainfo.in_unroll_factor/2))
        for vov,via in zip(vovecs, viarrs):
            code += '      %s += ' % vov
            for i,viv in enumerate(vivecs):
                if i: code += ' + '
                code += '%s[%s]*%s' % (via,i,viv)
            code += ';\n'
        code += '      %s;\n' % '; '.join(['%s+=%s' % (a,ainfo.in_unroll_factor/2) for a in viarrs])
        code += '      %s+=%s;\n' % (col_inds, ainfo.in_unroll_factor)
        code += '      j+=%s;\n' % ainfo.in_unroll_factor
        code += '    }\n'

        # vector values assignments
        code += '    %s;\n' % '; '.join(['%s+=j' % a for a in iarrs])
        code += '    %s %s;\n' % (ainfo.data_type, ','.join(vtovec_decls))
        code += '    %s %s;\n' % (ainfo.data_type, ','.join(vovec_stores))

        # the inner orio.main.cleanup loop
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

        # the epiloque of the outer orio.main.loop
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
            code += '    v2df *%sv=(v2df *)%s;\n' % (iarr, iarr)
            code += '    v2df %s;\n' % vovec_inits[0]

            # the inner cleanup-orio.main.loop
            code += '    register int j=0;\n'
            code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
            code += '      v2df %s;\n' % ','.join('%s={%s,%s}' % (vivecs[i], ivec_refs[2*i],
                                                                  ivec_refs[2*i+1])
                                                  for i in range(0, ainfo.in_unroll_factor/2))
            for vov,via in zip(vovecs[:1], viarrs[:1]):
                code += '      %s += ' % vov
                for i,viv in enumerate(vivecs):
                    if i: code += ' + '
                    code += '%s[%s]*%s' % (via,i,viv)
                code += ';\n'
            code += '      %s+=%s;\n' % (viarrs[0], ainfo.in_unroll_factor/2)
            code += '      %s+=%s;\n' % (col_inds, ainfo.in_unroll_factor)
            code += '      j+=%s;\n' % ainfo.in_unroll_factor
            code += '    }\n'

            # vector values assignments
            code += '    %s+=j;\n' % iarr
            code += '    %s %s;\n' % (ainfo.data_type, vtovec_decls[0])
            code += '    %s %s;\n' % (ainfo.data_type, vovec_stores[0])

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

    def __generateParSseSimdInodeCode(self):
        '''Generate an inode optimized code that is parallel and SSE simdized'''
        return self.__generateSseSimdInodeCode(True)

    def __generateSseSimdInodeCode(self,parallelize=False):
        '''Generate an inode optimized code that is sequential and SSE simdized'''
        
        # get the argument information
        ainfo = self.ainfo

        # start generating the optimized code
        code = ''
        code += '%s tbuf[%s];\n' % (ainfo.data_type, 2*ainfo.out_unroll_factor)

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
            code += ('#pragma omp parallel for shared(%s,%s,%s,%s,%s,%s,%s) private(n,tbuf)\n' %
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
        vovecs = ['%sv' % v for v in ovecs]
        vovec_inits = ['%s=_mm_set1_pd(%s)' % (r, ainfo.init_val) for r in vovecs]
        vovec_assgs = ['_mm_storer_pd(&tbuf[%s],%s)' % (2*i,r) for i,r in enumerate(vovecs)]
        vovec_stores = ['%s=tbuf[%s]+tbuf[%s]' % (v,2*i,2*i+1) for i,v in enumerate(ovecs)]
        viarrs = [['%sv%s' % (a,i) for i in range(ainfo.in_unroll_factor/2)] for a in iarrs]
        viarr_inits = [['%s=_mm_load_pd(&%s[%s])' % (r,a,2*i) for i,r in enumerate(rs)]
                       for rs,a in zip(viarrs, iarrs)]
        vivecs = ['%s%sv' % (ainfo.in_vector, i) for i in range(0, ainfo.in_unroll_factor/2)]
        
        # the outer orio.main.loop
        code += '  register int i=0;\n'
        code += '  while (i<=rlength-%s) {\n' % ainfo.out_unroll_factor
        if len(iarr_inits) > 0:
            code += '    %s %s;\n' % (ainfo.data_type, ','.join(map(lambda x:'*'+x, iarr_inits)))
        code += '    __m128d %s;\n' % ','.join(vovec_inits)

        # the inner orio.main.main loop
        code += '    register int j=0;\n'
        code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
        code += '      __m128d %s;\n' % ','.join(['%s=_mm_setr_pd(%s,%s)' % (vivecs[i],
                                                                             ivec_refs[2*i],
                                                                             ivec_refs[2*i+1])
                                                  for i in range(0, ainfo.in_unroll_factor/2)])
        for ins in viarr_inits:
            code += '      __m128d %s;\n' % ','.join(ins)
        for vov,vias in zip(vovecs, viarrs):
            for viv, via in zip(vivecs, vias):
                code += '      %s=_mm_add_pd(%s,_mm_mul_pd(%s,%s));\n' % (vov,vov,via,viv)
        code += '      %s;\n' % '; '.join(['%s+=%s' % (a,ainfo.in_unroll_factor) for a in iarrs])
        code += '      %s+=%s;\n' % (col_inds, ainfo.in_unroll_factor)
        code += '      j+=%s;\n' % ainfo.in_unroll_factor
        code += '    }\n'

        # vector values assignments
        code += '    %s;\n' % '; '.join(vovec_assgs)
        code += '    %s %s;\n' % (ainfo.data_type, ','.join(vovec_stores))

        # the inner orio.main.cleanup loop
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

        # the epiloque of the outer orio.main.loop
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
            code += '    __m128d %s;\n' % vovec_inits[0]

            # the inner cleanup-orio.main.loop
            code += '    register int j=0;\n'
            code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
            code += '      __m128d %s;\n' % ','.join(['%s=_mm_setr_pd(%s,%s)' % (vivecs[i],
                                                                                 ivec_refs[2*i],
                                                                                 ivec_refs[2*i+1])
                                                      for i in range(0, ainfo.in_unroll_factor/2)])
            code += '      __m128d %s;\n' % ','.join(viarr_inits[0])
            for vov,vias in zip(vovecs[:1], viarrs[:1]):
                for viv, via in zip(vivecs, vias):
                    code += '      %s=_mm_add_pd(%s,_mm_mul_pd(%s,%s));\n' % (vov,vov,via,viv)
            code += '      %s+=%s;\n' % (iarrs[0], ainfo.in_unroll_factor)
            code += '      %s+=%s;\n' % (col_inds, ainfo.in_unroll_factor)
            code += '      j+=%s;\n' % ainfo.in_unroll_factor
            code += '    }\n'

            # vector values assignments
            code += '    %s;\n' % vovec_assgs[0]
            code += '    %s %s;\n' % (ainfo.data_type, vovec_stores[0])

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

    def __generateParXlcSimdInodeCode(self):
        '''Generate an inode optimized code that is parallel and XLC simdized'''
        return self.__generateXlcSimdInodeCode(True)

    def __generateXlcSimdInodeCode(self,parallelize=False):
        '''Generate an inode optimized code that is sequential and XLC simdized'''

        # get the argument information
        ainfo = self.ainfo

        # start generating the optimized code
        code = ''
        code += '%s zerobuf[2]={%s,%s};\n' % (ainfo.data_type, ainfo.init_val, ainfo.init_val)
        code += '%s tbuf[%s];\n' % (ainfo.data_type, 2*ainfo.out_unroll_factor)
        code += '%s xbuf[%s];\n' % (ainfo.data_type, ainfo.in_unroll_factor)

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
            code += ('#pragma omp parallel for shared(%s,%s,%s,%s,%s,%s,%s,zerobuf) private(n,tbuf,xbuf)\n' %
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
        vovecs = ['%sv' % v for v in ovecs]
        vovec_inits = ['%s=__lfpd(zerobuf)' % (r) for r in vovecs]
        vovec_assgs = ['__stfpd(&tbuf[%s],%s)' % (2*i,r) for i,r in enumerate(vovecs)]
        vovec_stores = ['%s=tbuf[%s]+tbuf[%s]' % (v,2*i,2*i+1) for i,v in enumerate(ovecs)]
        viarrs = [['%sv%s' % (a,i) for i in range(ainfo.in_unroll_factor/2)] for a in iarrs]
        viarr_inits = [['%s=__lfpd(&%s[%s])' % (r,a,2*i) for i,r in enumerate(rs)]
                       for rs,a in zip(viarrs, iarrs)]
        vivecs = ['%s%sv' % (ainfo.in_vector, i) for i in range(0, ainfo.in_unroll_factor/2)]
        
        # the outer orio.main.loop
        code += '  register int i=0;\n'
        code += '  while (i<=rlength-%s) {\n' % ainfo.out_unroll_factor
        if len(iarr_inits) > 0:
            code += '    %s %s;\n' % (ainfo.data_type, ','.join(map(lambda x:'*'+x, iarr_inits)))
        code += '    double _Complex %s;\n' % ','.join(vovec_inits)

        # the inner orio.main.main loop
        code += '    register int j=0;\n'
        code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
        code += '      %s;\n' % '; '.join('xbuf[%s]=%s' % (i,ivec_refs[i])
                                          for i in range(ainfo.in_unroll_factor))
        code += '      double _Complex %s;\n' % ','.join(['%s=__lfpd(&xbuf[%s])' % (vivecs[i], 2*i)
                                                          for i in range(0,ainfo.in_unroll_factor/2)])
        for ins in viarr_inits:
            code += '      double _Complex %s;\n' % ','.join(ins)
        for vov,vias in zip(vovecs, viarrs):
            for viv, via in zip(vivecs, vias):
                code += '      %s=__fpmadd(%s,%s,%s);\n' % (vov,vov,via,viv)
        code += '      %s;\n' % '; '.join(['%s+=%s' % (a,ainfo.in_unroll_factor) for a in iarrs])
        code += '      %s+=%s;\n' % (col_inds, ainfo.in_unroll_factor)
        code += '      j+=%s;\n' % ainfo.in_unroll_factor
        code += '    }\n'

        # vector values assignments
        code += '    %s;\n' % '; '.join(vovec_assgs)
        code += '    %s %s;\n' % (ainfo.data_type, ','.join(vovec_stores))

        # the inner orio.main.cleanup loop
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

        # the epiloque of the outer orio.main.loop
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
            code += '    double _Complex %s;\n' % vovec_inits[0]

            # the inner cleanup-orio.main.loop
            code += '    register int j=0;\n'
            code += '    while (j<=clength-%s) {\n' % ainfo.in_unroll_factor
            code += '      %s;\n' % '; '.join('xbuf[%s]=%s' % (i,ivec_refs[i])
                                             for i in range(ainfo.in_unroll_factor))
            code += '      double _Complex %s;\n' % ','.join(['%s=__lfpd(&xbuf[%s])' % (vivecs[i],2*i)
                                                              for i in range(ainfo.in_unroll_factor/2)])
            code += '      double _Complex %s;\n' % ','.join(viarr_inits[0])
            for vov,vias in zip(vovecs[:1], viarrs[:1]):
                for viv, via in zip(vivecs, vias):
                    code += '      %s=__fpmadd(%s,%s,%s);\n' % (vov,vov,via,viv)
            code += '      %s+=%s;\n' % (iarrs[0], ainfo.in_unroll_factor)
            code += '      %s+=%s;\n' % (col_inds, ainfo.in_unroll_factor)
            code += '      j+=%s;\n' % ainfo.in_unroll_factor
            code += '    }\n'

            # vector values assignments
            code += '    %s;\n' % vovec_assgs[0]
            code += '    %s %s;\n' % (ainfo.data_type, vovec_stores[0])

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

    def __generateParCode(self):
        '''Generate an optimized code that is parallel'''
        return self.__generateCode(True)

    def __generateCode(self, parallelize=False):
        '''Generate an optimized code that is sequential'''

        # get the argument information
        ainfo = self.ainfo

        # generate the optimized code
        code = ''

        # the outer orio.main.loop
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

            # the inner orio.main.main loop
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

            # the inner orio.main.cleanup loop
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=lb%s-1) {\n' % (i+1)
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j+=1;\n'
                code += '  }\n'
            code += '  %s[i%s]=%s0;\n' % (ainfo.out_vector, '+%s'%i if i else '', ainfo.out_vector)

        # to close the outer orio.main.loop
        if not parallelize:
            code += '  i+=%s;\n' % ainfo.out_unroll_factor
        code += '} \n'

        # the outer cleanup loop
        if ainfo.out_unroll_factor > 1:
            if parallelize:
                code += 'i=lbound;\n'

            # the inner cleanup-orio.main.loop
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

    def __generateParGccSimdCode(self):
        '''Generate an optimized code that is parallel and GCC simdized'''
        return self.__generateGccSimdCode(True)

    def __generateGccSimdCode(self, parallelize=False):
        '''Generate an optimized code that is sequential and GCC simdized'''

        # get the argument information
        ainfo = self.ainfo

        # generate the optimized code
        code = ''

        # the outer orio.main.loop
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

            # the inner orio.main.main loop
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

            # the inner orio.main.cleanup loop
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=lb%s-1) {\n' % (i+1)
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j+=1;\n'
                code += '  }\n'
            code += '  %s[i%s]=%s0;\n' % (ainfo.out_vector, '+%s'%i if i else '', ainfo.out_vector)

        # to close the outer orio.main.loop
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

            # the inner cleanup-orio.main.loop
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

    def __generateParSseSimdCode(self):
        '''Generate an optimized code that is parallel and SSE simdized'''
        return self.__generateSseSimdCode(True)

    def __generateSseSimdCode(self, parallelize=False):
        '''Generate an optimized code that is sequential and SSE simdized'''

        # get the argument information
        ainfo = self.ainfo

        # generate the optimized code
        code = ''

        # the outer orio.main.loop
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
        code += '  %s tbuf[2];\n' % ainfo.data_type
        code += '  register int %s;\n' % ','.join(['lb%s=%s[i%s]' % (i, ainfo.row_inds,
                                                                     '+%s'%i if i else '')
                                                   for i in range(0, ainfo.out_unroll_factor+1)])
        
        # the unrolled inner loop
        for i in range(0, ainfo.out_unroll_factor):
            code += '  %s%s0v=_mm_set1_pd(%s);\n' % ('' if i else '__m128d ', ainfo.out_vector,
                                                     ainfo.init_val)

            # the inner orio.main.main loop
            code += '  %sj=lb%s;\n' % ('' if i else 'register int ', i)
            code += '  while (j<=lb%s-%s) {\n' % (i+1, ainfo.in_unroll_factor)
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '    __m128d '
                if j: code += ','
                code += ('%s%sv=_mm_setr_pd(%s[%s[j%s]],%s[%s[j%s]])' %
                         (ainfo.in_vector, j,
                          ainfo.in_vector, ainfo.col_inds, '+%s'%(2*j) if (2*j) else '',
                          ainfo.in_vector, ainfo.col_inds, '+%s'%(2*j+1) if (2*j+1) else '',))
            code += ';\n'
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '    __m128d '
                if j: code += ','
                code += ('%s%sv=_mm_setr_pd(%s[j%s],%s[j%s])' %
                         (ainfo.in_matrix, j,
                          ainfo.in_matrix, '+%s'%(2*j) if (2*j) else '',
                          ainfo.in_matrix, '+%s'%(2*j+1) if (2*j+1) else '',))
            code += ';\n'
            for j in range(0, ainfo.in_unroll_factor/2):
                code += ('    %s0v=_mm_add_pd(%s0v,_mm_mul_pd(%s%sv,%s%sv));\n' %
                         (ainfo.out_vector, ainfo.out_vector, ainfo.in_matrix, j, ainfo.in_vector, j))
            code += '    j+=%s; \n' % ainfo.in_unroll_factor
            code += '  }\n'

            # vector values assignments
            code += '  _mm_storer_pd(tbuf,%s0v);\n' % ainfo.out_vector
            code += '  %s%s0=tbuf[0]+tbuf[1];\n' % ('' if i else '%s '%ainfo.data_type,
                                                    ainfo.out_vector)
            
            # the inner orio.main.cleanup loop
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=lb%s-1) {\n' % (i+1)
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j+=1;\n'
                code += '  }\n'
            code += '  %s[i%s]=%s0;\n' % (ainfo.out_vector, '+%s'%i if i else '', ainfo.out_vector)

        # to close the outer orio.main.loop
        if not parallelize:
            code += '  i+=%s;\n' % ainfo.out_unroll_factor
        code += '} \n'

        # the outer cleanup loop
        if ainfo.out_unroll_factor > 1:
            code += '%s tbuf[2];\n' % ainfo.data_type
            if parallelize:
                code += 'i=lbound;\n'
            code += 'while (i<=%s-1) {\n' % ainfo.total_rows
            code += '  __m128d %s0v=_mm_set1_pd(%s);\n' % (ainfo.out_vector, ainfo.init_val)
            code += '  register int j=%s[i], ub=%s[i+1];\n' % (ainfo.row_inds, ainfo.row_inds)

            # the inner cleanup-orio.main.loop
            code += '  while (j<=ub-%s) {\n' % ainfo.in_unroll_factor
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '    __m128d '
                if j: code += ','
                code += ('%s%sv=_mm_setr_pd(%s[%s[j%s]],%s[%s[j%s]])' %
                         (ainfo.in_vector, j,
                          ainfo.in_vector, ainfo.col_inds, '+%s'%(2*j) if (2*j) else '',
                          ainfo.in_vector, ainfo.col_inds, '+%s'%(2*j+1) if (2*j+1) else '',))
            code += ';\n'
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '    __m128d '
                if j: code += ','
                code += ('%s%sv=_mm_setr_pd(%s[j%s],%s[j%s])' %
                         (ainfo.in_matrix, j,
                          ainfo.in_matrix, '+%s'%(2*j) if (2*j) else '',
                          ainfo.in_matrix, '+%s'%(2*j+1) if (2*j+1) else '',))
            code += ';\n'
            for j in range(0, ainfo.in_unroll_factor/2):
                code += ('    %s0v=_mm_add_pd(%s0v,_mm_mul_pd(%s%sv,%s%sv));\n' %
                         (ainfo.out_vector, ainfo.out_vector, ainfo.in_matrix, j, ainfo.in_vector, j))
            code += '    j+=%s;\n' % ainfo.in_unroll_factor
            code += '  }\n'

            # vector values assignments
            code += '  _mm_storer_pd(tbuf,%s0v);\n' % ainfo.out_vector
            code += '  %s %s0=tbuf[0]+tbuf[1];\n' % (ainfo.data_type, ainfo.out_vector)

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

    def __generateParXlcSimdCode(self):
        '''Generate an optimized code that is parallel and XLC simdized'''
        return self.__generateXlcSimdCode(True)

    def __generateXlcSimdCode(self, parallelize=False):
        '''Generate an optimized code that is sequential and XLC simdized'''

        # get the argument information
        ainfo = self.ainfo

        # generate the optimized code
        code = ''
        code += '%s zerobuf[2]={%s,%s};\n' % (ainfo.data_type, ainfo.init_val, ainfo.init_val)
        code += '%s tbuf[2];\n' % ainfo.data_type
        code += '%s xbuf[%s];\n' % (ainfo.data_type, ainfo.in_unroll_factor)

        # the outer orio.main.loop
        if parallelize:
            code += 'register int i;\n'
            if ainfo.out_unroll_factor==1:
                code += 'register int lbound=%s;\n' % ainfo.total_rows
            else:
                code += 'register int lbound=%s-(%s%%%s);\n' % (ainfo.total_rows, ainfo.total_rows,
                                                                ainfo.out_unroll_factor)
            code += 'omp_set_num_threads(%s);\n' % ainfo.num_threads
            code += ('#pragma omp parallel for shared(%s,%s,%s,%s,%s,%s,lbound,zerobuf) private(i,tbuf,xbuf)\n' %
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
            
            # the inner orio.main.main loop
            code += '  %sj=lb%s;\n' % ('' if i else 'register int ', i)

            # simdization (aligned case)
            code += '  if ((((int) &%s[j]) & 0xf) == 0) {\n' % ainfo.in_matrix
            code += '    double _Complex %s0v=__lfpd(zerobuf);\n' % ainfo.out_vector
            code += '    while (j<=lb%s-%s) {\n' % (i+1, ainfo.in_unroll_factor)
            code += '      %s;\n' % '; '.join('xbuf[%s]=%s[%s[j%s]]' % (i, ainfo.in_vector,
                                                                      ainfo.col_inds,
                                                                      '+%s'%i if i else '')
                                           for i in range(ainfo.in_unroll_factor))
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '      double _Complex '
                if j: code += ','
                code += '%s%sv=__lfpd(&xbuf[%s])' % (ainfo.in_vector, j, 2*j)
            code += ';\n'
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '      double _Complex '
                if j: code += ','
                code += '%s%sv=__lfpd(&%s[j%s])' % (ainfo.in_matrix, j, ainfo.in_matrix,
                                                    '+%s'%(2*j) if (2*j) else '')
            code += ';\n'
            for j in range(0, ainfo.in_unroll_factor/2):
                code += ('      %s0v=__fpmadd(%s0v,%s%sv,%s%sv);\n' %
                         (ainfo.out_vector, ainfo.out_vector, ainfo.in_matrix, j, ainfo.in_vector, j))
            code += '      j+=%s; \n' % ainfo.in_unroll_factor
            code += '    }\n'
            code += '    __stfpd(tbuf,%s0v);\n' % ainfo.out_vector
            code += '    %s0=tbuf[0]+tbuf[1];\n' % ainfo.out_vector

            # unaligned case
            code += '  } else {\n'
            code += '    while (j<=lb%s-%s) {\n' % (i+1, ainfo.in_unroll_factor)
            code += '      %s0 += ' % ainfo.out_vector
            for j in range(0, ainfo.in_unroll_factor):
                if j: code += ' + '
                code += '%s[j%s]*%s[%s[j%s]]' % (ainfo.in_matrix, '+%s'%j if j else '',
                                                 ainfo.in_vector, ainfo.col_inds,
                                                 '+%s'%j if j else '')
            code += ';\n'
            code += '      j+=%s;\n' % ainfo.in_unroll_factor
            code += '    }\n'
            code += '  }\n'

            # the inner orio.main.cleanup loop
            if ainfo.in_unroll_factor > 1:
                code += '  while (j<=lb%s-1) {\n' % (i+1)
                code += '    %s0 += %s[j]*%s[%s[j]];\n' % (ainfo.out_vector, ainfo.in_matrix,
                                                           ainfo.in_vector, ainfo.col_inds)
                code += '    j+=1;\n'
                code += '  }\n'
            code += '  %s[i%s]=%s0;\n' % (ainfo.out_vector, '+%s'%i if i else '', ainfo.out_vector)

        # to close the outer orio.main.loop
        if not parallelize:
            code += '  i+=%s;\n' % ainfo.out_unroll_factor
        code += '} \n'

        # the outer cleanup loop
        if ainfo.out_unroll_factor > 1:
            if parallelize:
                code += 'i=lbound;\n'
            code += 'while (i<=%s-1) {\n' % ainfo.total_rows
            code += '  %s %s0=%s;\n' % (ainfo.data_type, ainfo.out_vector, ainfo.init_val)

            # the inner cleanup-orio.main.loop
            code += '  register int j=%s[i], ub=%s[i+1];\n' % (ainfo.row_inds, ainfo.row_inds)

            # simdization (aligned case)
            code += '  if ((((int) &%s[j]) & 0xf) == 0) {\n' % ainfo.in_matrix
            code += '    double _Complex %s0v=__lfpd(zerobuf);\n' % ainfo.out_vector
            code += '    while (j<=ub-%s) {\n' % ainfo.in_unroll_factor
            code += '      %s;\n' % '; '.join('xbuf[%s]=%s[%s[j%s]]' % (i, ainfo.in_vector,
                                                                        ainfo.col_inds,
                                                                        '+%s'%i if i else '')
                                              for i in range(ainfo.in_unroll_factor))
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '      double _Complex '
                if j: code += ','
                code += '%s%sv=__lfpd(&xbuf[%s])' % (ainfo.in_vector, j, 2*j)
            code += ';\n'
            for j in range(0, ainfo.in_unroll_factor/2):
                if not j: code += '      double _Complex '
                if j: code += ','
                code += '%s%sv=__lfpd(&%s[j%s])' % (ainfo.in_matrix, j, ainfo.in_matrix,
                                                    '+%s'%(2*j) if (2*j) else '')
            code += ';\n'
            for j in range(0, ainfo.in_unroll_factor/2):
                code += ('      %s0v=__fpmadd(%s0v,%s%sv,%s%sv);\n' %
                         (ainfo.out_vector, ainfo.out_vector, ainfo.in_matrix, j, ainfo.in_vector, j))
            code += '      j+=%s;\n' % ainfo.in_unroll_factor
            code += '    }\n'
            code += '    __stfpd(tbuf,%s0v);\n' % ainfo.out_vector
            code += '    %s0=tbuf[0]+tbuf[1];\n' % ainfo.out_vector

            # unaligned case
            code += '  } else {\n'
            code += '    while (j<=ub-%s) {\n' % ainfo.in_unroll_factor
            code += '      %s0 += ' % ainfo.out_vector
            for j in range(0, ainfo.in_unroll_factor):
                if j: code += ' + '
                code += '%s[j%s]*%s[%s[j%s]]' % (ainfo.in_matrix, '+%s'%j if j else '',
                                                 ainfo.in_vector, ainfo.col_inds,
                                                 '+%s'%j if j else '')
            code += ';\n'
            code += '      j+=%s;\n' % ainfo.in_unroll_factor
            code += '    }\n'
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

    def generate(self):
        '''To generate optimized SpMV code'''

        # the argument information
        ainfo = self.ainfo

        # generate the optimized code
        # no structure format
        if ainfo.block_structure == arg_info.ArgInfo.BSTRUC_NONE:
            if ainfo.num_threads == 1:
                if ainfo.simd == arg_info.ArgInfo.SIMD_NONE:
                    code = self.__generateCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_GCC:
                    code = self.__generateGccSimdCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_SSE:
                    code = self.__generateSseSimdCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_XLC:
                    code = self.__generateXlcSimdCode()
                else:
                    err('orio.module.spmv.codegen: SpMV: unsupported SIMD type')
            else:
                if ainfo.simd == arg_info.ArgInfo.SIMD_NONE:
                    code = self.__generateParCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_GCC:
                    code = self.__generateParGccSimdCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_SSE:
                    code = self.__generateParSseSimdCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_XLC:
                    code = self.__generateParXlcSimdCode()
                else:
                    err('orio.module.spmv.codegen: SpMV: unsupported SIMD type')

        # inode structure format
        elif ainfo.block_structure == arg_info.ArgInfo.BSTRUC_INODE:
            if ainfo.num_threads == 1:
                if ainfo.simd == arg_info.ArgInfo.SIMD_NONE:
                    code = self.__generateInodeCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_GCC:
                    code = self.__generateGccSimdInodeCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_SSE:
                    code = self.__generateSseSimdInodeCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_XLC:
                    code = self.__generateXlcSimdInodeCode()
                else:
                    err('orio.module.spmv.codegen: SpMV: unsupported SIMD type')
            else:
                if ainfo.simd == arg_info.ArgInfo.SIMD_NONE:
                    code = self.__generateParInodeCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_GCC:
                    code = self.__generateParGccSimdInodeCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_SSE:
                    code = self.__generateParSseSimdInodeCode()
                elif ainfo.simd == arg_info.ArgInfo.SIMD_XLC:
                    code = self.__generateParXlcSimdInodeCode()
                else:
                    err('orio.module.spmv.codegen: SpMV: unsupported SIMD type')
        else:
            err('orio.module.spmv.codegen: SpMV: unsupported matrix block structure')

        # return the optimized code
        return code

