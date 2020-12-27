#
# Contain the transformation procedure
#

import orio.module.loop.ast, orio.module.loop.ast_lib.common_lib, orio.module.loop.ast_lib.constant_folder
import orio.module.loop.ast_lib.forloop_lib
from orio.main.util.globals import *
from functools import reduce

#-----------------------------------------

class Transformation:
    '''Code transformation interface'''

    def __init__(self, aref, suffix, dtype, dimsizes, stmt):
        '''To instantiate a code transformation object'''

        # remove any whitespaces from the array reference string
        aref = aref.replace(' ','')
        
        self.aref = aref
        self.suffix = '_buf'
        if suffix != None:
            self.suffix = suffix
        self.dtype = 'double'
        if dtype != None:
            self.dtype = dtype
        self.dimsizes = dimsizes
        self.stmt = stmt
        
        self.flib = orio.module.loop.ast_lib.forloop_lib.ForLoopLib()
        self.clib = orio.module.loop.ast_lib.common_lib.CommonLib()
        self.cfolder = orio.module.loop.ast_lib.constant_folder.ConstFolder()

    #----------------------------------------------------------

    def transform(self):
        '''Perform the code transformation.'''

        # get the array reference information
        aref_info = self.__getARefInfo()

        # check if one of the array buffer dimension sizes is one (i.e. no tiling)
        aref, arr_name, ivar_names, dim_sizes, is_output = aref_info
        one_one = reduce(lambda x,y: x or y, [x==1 for x in dim_sizes], False)
        if one_one:
            decl = orio.module.loop.ast.VarDecl(self.dtype, [arr_name + self.suffix])
            if isinstance(self.stmt, orio.module.loop.ast.CompStmt):
                self.stmt.stmts = [decl] + self.stmt.stmts
            else:
                self.stmt = orio.module.loop.ast.CompStmt([decl, self.stmt])
            return self.stmt

        # perform array copy optimization
        tstmt, is_done, decl, rinfo = self.__optimizeCopy(self.stmt, aref_info, [])

        # if it is done
        if is_done:
            if decl == None:
                err('orio.module.loop.submodule.arrcopy.transformation:  array copy optimization was unsuccessful')
            else:
                if isinstance(tstmt, orio.module.loop.ast.CompStmt):
                    tstmt.stmts = [decl] + tstmt.stmts
                else:
                    tstmt = orio.module.loop.ast.CompStmt([decl, tstmt])
        else:
            err('orio.module.loop.submodule.arrcopy.transformation:  array copy optimization must be applied on a tiled perfect loop nest')
        
        # return the transformed statement
        return tstmt

    #----------------------------------------------------------
    # Private methods
    #----------------------------------------------------------

    def __containARef(self, tnode, is_output):
        '''To determine if the given expression contains the specified array reference'''

        if tnode == None:
            return []

        if isinstance(tnode, orio.module.loop.ast.NumLitExp):
            return []

        elif isinstance(tnode, orio.module.loop.ast.StringLitExp):
            return []

        elif isinstance(tnode, orio.module.loop.ast.IdentExp):
            return []

        elif isinstance(tnode, orio.module.loop.ast.ArrayRefExp):
            cur_aref = str(tnode).replace(' ','')
            if cur_aref == self.aref:
                return [(tnode, is_output)]
            else:
                return []

        elif isinstance(tnode, orio.module.loop.ast.FunCallExp):
            result = self.__containARef(tnode.exp, is_output)
            return result + reduce(lambda x,y: x+y,
                                   [self.__containARef(a, is_output) for a in tnode.args],
                                   [])
        
        elif isinstance(tnode, orio.module.loop.ast.UnaryExp):
            return self.__containARef(tnode.exp, is_output)

        elif isinstance(tnode, orio.module.loop.ast.BinOpExp):
            if tnode.op_type == orio.module.loop.ast.BinOpExp.EQ_ASGN:
                result = self.__containARef(tnode.lhs, True)
            else:
                result = self.__containARef(tnode.lhs, False)
            return result + self.__containARef(tnode.rhs, False)

        elif isinstance(tnode, orio.module.loop.ast.ParenthExp):
            return self.__containARef(tnode.exp, is_output)

        elif isinstance(tnode, orio.module.loop.ast.ExpStmt):
            return self.__containARef(tnode.exp, is_output)
    
        elif isinstance(tnode, orio.module.loop.ast.CompStmt):
            return reduce(lambda x,y: x+y,
                          [self.__containARef(s, is_output) for s in tnode.stmts],
                          [])

        elif isinstance(tnode, orio.module.loop.ast.IfStmt):
            return (self.__containARef(tnode.test, is_output) +
                    self.__containARef(tnode.true_stmt, is_output) +
                    self.__containARef(tnode.false_stmt, is_output))
        
        elif isinstance(tnode, orio.module.loop.ast.ForStmt):
            return self.__containARef(tnode.stmt, is_output) 
    
        elif isinstance(tnode, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.arrcopy.transformation internal error: unprocessed transform statement')
            
        elif isinstance(tnode, orio.module.loop.ast.NewAST):
            return []
        
        elif isinstance(tnode, orio.module.loop.ast.Comment):
            return tnode

        else:
            err('orio.module.loop.submodule.arrcopy.transformation internal error: unexpected AST type: "%s"' % tnode.__class__.__name__)

    #----------------------------------------------------------
    
    def __replaceARef(self, tnode, aref_str, replacement):
        '''To replace the given array reference with the specified replacement'''

        if tnode == None:
            return None

        if isinstance(tnode, orio.module.loop.ast.NumLitExp):
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.StringLitExp):
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.IdentExp):
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.ArrayRefExp):
            if str(tnode) == aref_str:
                return replacement.replicate()
            else:
                return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.FunCallExp):
            tnode.exp = self.__replaceARef(tnode.exp, aref_str, replacement)
            tnode.args = [self.__replaceARef(a, aref_str, replacement) for a in tnode.args]
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.UnaryExp):
            tnode.exp = self.__replaceARef(tnode.exp, aref_str, replacement)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.BinOpExp):
            tnode.lhs = self.__replaceARef(tnode.lhs, aref_str, replacement)
            tnode.rhs = self.__replaceARef(tnode.rhs, aref_str, replacement)
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.ParenthExp):
            tnode.exp = self.__replaceARef(tnode.exp, aref_str, replacement)
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.ExpStmt):
            tnode.exp = self.__replaceARef(tnode.exp, aref_str, replacement)
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.CompStmt):
            tnode.stmts = [self.__replaceARef(s, aref_str, replacement) for s in tnode.stmts]
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.IfStmt):
            tnode.test = self.__replaceARef(tnode.test, aref_str, replacement)
            tnode.true_stmt = self.__replaceARef(tnode.true_stmt, aref_str, replacement)
            tnode.false_stmt = self.__replaceARef(tnode.false_stmt, aref_str, replacement)
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.ForStmt):
            tnode.stmt = self.__replaceARef(tnode.stmt, aref_str, replacement)
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.arrcopy.transformation internal error: unprocessed transform statement')
            
        elif isinstance(tnode, orio.module.loop.ast.NewAST):
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.Comment):
            return tnode
        
        else:
            err('orio.module.loop.submodule.arrcopy.transformation internal error: unexpected AST type: "%s"' % tnode.__class__.__name__)
            
    #----------------------------------------------------------

    def __createArrCopy(self, aref, arr_name, loop_headers, buf_dsizes):
        '''
        To create the array copying optimization (i.e. declaration, the copying loop,
        the storing loop, the reference to the array copy)
        '''
        
        # create the name of the array buffer
        intmd_name = arr_name + self.suffix

        # generate the declaration of the array buffer
        arr_dname = intmd_name
        if len(buf_dsizes) > 0:
            arr_dname += '[' + ']['.join(map(str, buf_dsizes)) + ']'
        decl = orio.module.loop.ast.VarDecl(self.dtype, [arr_dname])

        # generate the reference to the array buffer: X_buffer[(i-LBi)/STi][...]
        intmd_aref = orio.module.loop.ast.IdentExp(intmd_name)
        for index_id, lbound_exp, ubound_exp, stride_exp in loop_headers:
            sub_exp = orio.module.loop.ast.BinOpExp(index_id.replicate(),
                                               lbound_exp.replicate(),
                                               orio.module.loop.ast.BinOpExp.SUB)
            sub_exp = orio.module.loop.ast.ParenthExp(sub_exp)
            sub_exp = orio.module.loop.ast.BinOpExp(sub_exp,
                                               stride_exp.replicate(),
                                               orio.module.loop.ast.BinOpExp.DIV)
            sub_exp = self.cfolder.fold(sub_exp)
            intmd_aref = orio.module.loop.ast.ArrayRefExp(intmd_aref, sub_exp)

        # generate the copying loop: <for-loops> X_buffer[(i-LBi)/STi][...] = X_orig[...][...];
        rev_loop_headers = loop_headers[:]
        rev_loop_headers.reverse()
        copy_loop = orio.module.loop.ast.BinOpExp(intmd_aref.replicate(),
                                             aref.replicate(),
                                             orio.module.loop.ast.BinOpExp.EQ_ASGN)
        copy_loop = orio.module.loop.ast.ExpStmt(copy_loop)
        for index_id, lbound_exp, ubound_exp, stride_exp in rev_loop_headers:
            copy_loop = self.flib.createForLoop(index_id, lbound_exp, ubound_exp,
                                                stride_exp, copy_loop)
        copy_loop = orio.module.loop.ast.Container(copy_loop)
        
        # generate the storing loop: <for-loops> X_orig[...][...] = X_buffer[(i-LBi)/STi][...];
        store_loop = orio.module.loop.ast.BinOpExp(aref.replicate(),
                                              intmd_aref.replicate(),
                                              orio.module.loop.ast.BinOpExp.EQ_ASGN)
        store_loop = orio.module.loop.ast.ExpStmt(store_loop)
        for index_id, lbound_exp, ubound_exp, stride_exp in rev_loop_headers:
            store_loop = self.flib.createForLoop(index_id, lbound_exp, ubound_exp,
                                                 stride_exp, store_loop)
        store_loop = orio.module.loop.ast.Container(store_loop)

        # return all resulting information
        return (decl, intmd_aref, copy_loop, store_loop)
        
    #----------------------------------------------------------

    def __optimizeCopy(self, stmt, aref_info, outer_lids):
        '''To perform array copy optimization'''

        if stmt == None:
            return (None, True, None, None)

        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            return (stmt, True, None, None)

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            nstmts = []
            ndecl = None
            nrinfo = None
            for s in stmt.stmts:
                cstmt, is_done, decl, rinfo = self.__optimizeCopy(s, aref_info, outer_lids)
                nstmts.append(cstmt)
                if is_done:
                    if decl != None:
                        if ndecl != None:
                            err('orio.module.loop.submodule.arrcopy.transformation: array copy optimization cannot work for imperfect ' +
                                   'loop nests')
                        else:
                            ndecl = decl
                    else:
                        pass
                else:
                    if nrinfo != None:
                        err('orio.module.loop.submodule.arrcopy.transformation:  array copy optimization cannot work for imperfect loop nests')
                    else:
                        nrinfo = rinfo
            stmt.stmts = nstmts
            if ndecl != None:
                if nrinfo != None:
                        err('orio.module.loop.submodule.arrcopy.transformation:  array copy optimization cannot work for imperfect loop nests')
                return (stmt, True, ndecl, None)
            else:
                if nrinfo != None:
                    return (stmt, False, None, nrinfo)
                else:
                    return (stmt, True, None, None)

        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            (stmt1, is_done1, decl1, rinfo1) = self.__optimizeCopy(stmt.true_stmt,
                                                                   aref_info, outer_lids)
            (stmt2, is_done2, decl2, rinfo2) = self.__optimizeCopy(stmt.false_stmt,
                                                                   aref_info, outer_lids)
            stmt.true_stmt = stmt1
            stmt.false_stmt = stmt2
            if is_done1 and is_done2:
                if decl1 != None and decl2 != None:
                    err('orio.module.loop.submodule.arrcopy.transformation:  array copy optimization cannot work for imperfect loop nests')
                return (stmt, True, (decl1 or decl2), None)
            elif is_done1:
                if decl1 != None:
                    err('orio.module.loop.submodule.arrcopy.transformation:  array copy optimization cannot work for imperfect loop nests')
                return (stmt, False, None, rinfo2)
            elif is_done2:
                if decl2 != None:
                    err('orio.module.loop.submodule.arrcopy.transformation:  array copy optimization cannot work for imperfect loop nests')
                return (stmt, False, None, rinfo1)
            else:
                err('orio.module.loop.submodule.arrcopy.transformation:  array copy optimization cannot work for imperfect loop nests')

        elif isinstance(stmt, orio.module.loop.ast.ForStmt):

            # unpack the array reference info
            aref, arr_name, ivar_names, dim_sizes, is_output = aref_info

            # get the loop structure 
            for_loop_info = self.flib.extractForLoopInfo(stmt)
            index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info

            # update the outer loop index names
            outer_lids = outer_lids[:] + [index_id.name]

            # if all the iteration variable names are subsumed by the outer loop index names
            if set(ivar_names) <= set(outer_lids):

                # check if the loop body contains the array reference to be optimized
                if not self.__containARef(stmt.stmt, False):
                    return (stmt, True, None, None)

                # move to outside the loop
                rem_inames = [i for i in ivar_names if i != index_id.name]
                ind_inames = [i for i in outer_lids if (self.clib.containIdentName(lbound_exp, i) or
                                               self.clib.containIdentName(ubound_exp, i) or
                                               self.clib.containIdentName(stride_exp, i))]
                loop_headers = [(index_id, lbound_exp, ubound_exp, stride_exp)]
                ipos = ivar_names.index(index_id.name)
                buf_dsizes = [dim_sizes[ipos]]
                rinfo = (rem_inames, ind_inames, loop_headers, buf_dsizes)
                return (stmt, False, None, rinfo)

            # otherwise
            else:
                
                # apply recursion on the loop body
                (nstmt, is_done, decl, rinfo) = self.__optimizeCopy(stmt.stmt, aref_info, outer_lids)
                
                # update the loop body
                stmt.stmt = nstmt

                # if the loop body has been transformed
                if is_done:
                    return (stmt, True, decl, None)

                # unpack the result info
                rem_inames, ind_inames, loop_headers, buf_dsizes = rinfo

                # if need to perform array copying now
                if index_id.name in ind_inames:
                    (decl, intmd_aref,
                     copy_loop, store_loop) = self.__createArrCopy(aref, arr_name,
                                                                   loop_headers, buf_dsizes)
                    prologue = [copy_loop]
                    epilogue = [store_loop] if is_output else [] 
                    stmt.stmt = self.__replaceARef(stmt.stmt, str(aref), intmd_aref)
                    if isinstance(stmt.stmt, orio.module.loop.ast.CompStmt):
                        stmt.stmt.stmts = prologue + stmt.stmt.stmts + epilogue
                    else:
                        stmt.stmt = orio.module.loop.ast.CompStmt(prologue + [stmt.stmt] + epilogue)
                    return (stmt, True, decl, None)
                
                # move to outside the loop
                if index_id.name in rem_inames:
                    nrem_inames = [i for i in rem_inames if i != index_id.name]
                    nind_inames = [i for i in outer_lids if (self.clib.containIdentName(lbound_exp, i) or
                                                    self.clib.containIdentName(ubound_exp, i) or
                                                    self.clib.containIdentName(stride_exp, i))]
                    nind_inames += ind_inames
                    nloop_headers = loop_headers[:]
                    nloop_headers.insert(0, (index_id, lbound_exp, ubound_exp, stride_exp))
                    ipos = ivar_names.index(index_id.name)
                    nbuf_dsizes = buf_dsizes[:]
                    nbuf_dsizes.insert(0, dim_sizes[ipos])
                    nrinfo = (nrem_inames, nind_inames, nloop_headers, nbuf_dsizes)
                    return (stmt, False, None, nrinfo)

                # otherwise
                return (stmt, False, None, rinfo)
                    
        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.arrcopy.transformation internal error: unprocessed transform statement')
            
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return (stmt, True, None, None)
        
        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return (stmt, True, None, None)

        else:
            err('orio.module.loop.submodule.arrcopy.transformation internal error: unexpected AST type: "%s"' % stmt.__class__.__name__)
        
    #----------------------------------------------------------

    def __getARefInfo(self):
        '''Return information about the array reference'''

        # get the actual array reference
        result = self.__containARef(self.stmt, False)
        aref = None
        is_output = False
        for a, i in result:
            aref = a
            is_output = is_output or i
        if aref == None:
            err('orio.module.loop.submodule.arrcopy.transformation:  array-copy statement does not contain array reference: "%s"' % self.aref)

        # get the array name and the array dimension expressions
        dexps = []
        exps = [aref]
        while len(exps) > 0:
            e = exps.pop(0)
            if isinstance(e, orio.module.loop.ast.ArrayRefExp):
                exps.insert(0, e.sub_exp)
                exps.insert(0, e.exp)
            else:
                dexps.append(e)
        arr_name = dexps.pop(0).name
        dim_exps = dexps

        # get the iteration variables used in the array reference
        all_lids = self.flib.getLoopIndexNames(self.stmt)
        ivar_names = []
        for e in dim_exps:
            c_inames = [i for i in all_lids if self.clib.containIdentName(e, i)]
            if len(c_inames) != 1:
                err(('orio.module.loop.submodule.arrcopy.transformation: each dimension expression of array reference %s must contain ' +
                        'exactly one iteration variable') % self.aref)
            iname = c_inames[0]
            if iname in ivar_names:
                err('orio.module.loop.submodule.arrcopy.transformation: repeated iteration variable name "%s" in array reference %s' %
                       (iname, aref))
            ivar_names.append(iname)

        # get the sizes of the dimensions of the array buffer
        if len(self.dimsizes) != len(dim_exps):
            err('orio.module.loop.submodule.arrcopy.transformation:  incorrect the number of array dimensions: %s' % self.dimsizes)
        dim_sizes = self.dimsizes
        
        # create information about the array reference
        aref_info = (aref, arr_name, ivar_names, dim_sizes, is_output)

        # return the information of the array reference
        return aref_info
        
