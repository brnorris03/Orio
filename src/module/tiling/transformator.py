#
# The implementation of the code transformator that performs loop tiling
#

import sys
import ast, ast_util

#-------------------------------------------------

class SimpleLoops:
    '''A simple abstraction used to represent a sequence of loops'''

    def __init__(self):
        '''
        To instantiate a sequence of loops.
        For instance, the following loop:
         for i
           S1
           for j
             for k
               S2
           S3
        is internally represented as follows:
         ('i', [S1, ('j', [('k', [S2])]), S3])
        '''

        self.loops = []
        self.ast_util = ast_util.ASTUtil()

    #----------------------------------------------

    def insertLoop(self, iter_names_seq, stmt):
        '''
        To add the given loop into this sequence of loops. Loop fusion is applied whenever possible.
        The given loop is represented with the sequence of the iterator names and
        the enclosed statement.
        '''

        # check if the given loop can be fused into one of the existing loops
        loops = self.loops
        inames_seq = iter_names_seq[:]
        is_done = False
        while inames_seq:
            iname = inames_seq[0]
            matching_loop = None
            for l in loops:
                if isinstance(l, tuple) and l[0] == iname:
                    matching_loop = l
            if matching_loop == None:
                break
            inames_seq.pop(0)
            if inames_seq:
                loops = matching_loop[1]
            else:
                matching_loop[1].append(stmt)
                is_done = True
                break

        # create a new loop and add it into the sequence of loops 
        if not is_done:
            inames_seq.reverse()
            l = stmt
            for iname in inames_seq:
                l = (iname, [l])
            loops.append(l)

    #----------------------------------------------

    def convertToASTs(self, loop_info_table, node = None):
        '''To generate a sequence of ASTs that correspond to this sequence of loops'''

        if node == None:
            node = self.loops

        if isinstance(node, tuple):
            iname, subnodes = node
            tsize_name, lb_exp, st_exp = loop_info_table[iname]
            id = ast.IdentExp(iname)
            lb = lb_exp
            tmp = ast.BinOpExp(ast.IdentExp(tsize_name), ast.ParenthExp(st_exp), ast.BinOpExp.SUB)
            ub = ast.BinOpExp(lb_exp, ast.ParenthExp(tmp), ast.BinOpExp.ADD)
            st = st_exp
            lbody = ast.CompStmt(self.convertToASTs(loop_info_table, subnodes))
            return self.ast_util.createForLoop(id, lb, ub, st, lbody)

        elif isinstance(node, list):
            return [self.convertToASTs(loop_info_table, n) for n in node]

        else:
            return node

#-------------------------------------------------

class Transformator:
    '''The code transformator that performs loop tiling'''

    def __init__(self, perf_params, tiling_info):
        '''To instantiate a code transformator'''

        num_level, tiling_table = tiling_info
        tiled_loop_inames = tiling_table.keys()
        tile_size_table = tiling_table

        self.perf_params = perf_params
        self.num_level = num_level
        self.tiled_loop_inames = tiled_loop_inames
        self.tile_size_table = tile_size_table
        self.ast_util = ast_util.ASTUtil()
        self.counter = 1

    #----------------------------------------------

    def __getIterName(self, iter_name, level):
        return iter_name + ('t%s' % level)

    def __getTileSizeName(self, iter_name, level):
        return ('T%s' % level) + iter_name

    def __getLoopBoundNames(self):
        lb_name = 'lb%s' % self.counter
        ub_name = 'ub%s' % self.counter
        self.counter += 1
        return (lb_name, ub_name)
    
    #----------------------------------------------

    def __getInterTileLoop(self, iter_name, tsize_name, lb_exp, ub_exp, st_exp, lbody):
        '''
        Generate an inter-tile loop:
          for (i=lb; i<=ub-(Ti-st); i+=Ti)
             lbody
        '''

        id = ast.IdentExp(iter_name)
        lb = lb_exp
        tmp = ast.BinOpExp(ast.IdentExp(tsize_name), ast.ParenthExp(st_exp), ast.BinOpExp.SUB)
        ub = ast.BinOpExp(ub_exp, ast.ParenthExp(tmp), ast.BinOpExp.SUB)
        st = ast.IdentExp(tsize_name)
        return self.ast_util.createForLoop(id, lb, ub, st, lbody)

    def __getIntraTileLoop(self, iter_name, tsize_name, lb_exp, st_exp, lbody):
        '''
        Generate an intra-tile loop:
          for (i=lb; i<=lb+(Ti-st); i+=st)
             lbody
        '''

        id = ast.IdentExp(iter_name)
        lb = lb_exp
        tmp = ast.BinOpExp(ast.IdentExp(tsize_name), ast.ParenthExp(st_exp), ast.BinOpExp.SUB)
        ub = ast.BinOpExp(lb_exp, ast.ParenthExp(tmp), ast.BinOpExp.ADD)
        st = st_exp
        return self.ast_util.createForLoop(id, lb, ub, st, lbody)

    #----------------------------------------------

    def __getMultiLevelTileLoop(self, num_level, iter_names, st_exps, lbody):
        '''
        Generate a multilevel-tile loop (for instance, suppose that the given iterator names is (i,j)
        and the given number of tiling levels is 2):
          for (it1=it2; it1<=it2+(T2i-T1i); it1+=T1i)
           for (jt1=jt2; jt1<=jt2+(T2j-T1j); jt1+=T1j)
            for (i=it1; i<=it1+(T1i-sti); i+=sti)
             for (j=jt1; j<=jt1+(T1j-stj); j+=stj)
               lbody          
        '''

        iter_names = iter_names[:]
        iter_names.reverse()
        st_exps = st_exps[:]
        st_exps.reverse()
        loop = lbody
        for level in range(1, num_level+1):
            if level == 1:
                for iname, st_exp in zip(iter_names, st_exps):
                    n_tsize_name = self.__getTileSizeName(iname, level)
                    lb = ast.IdentExp(self.__getIterName(iname, level))
                    loop = self.__getIntraTileLoop(iname, n_tsize_name, lb, st_exp, loop)
            else:
                for iname in iter_names:
                    c_iname = self.__getIterName(iname, level-1)
                    n_tsize_name = self.__getTileSizeName(iname, level)
                    lb = ast.IdentExp(self.__getIterName(iname, level))
                    st = ast.IdentExp(self.__getTileSizeName(iname, level-1))
                    loop = self.__getIntraTileLoop(c_iname, n_tsize_name, lb, st, loop)
        return loop
    
    #----------------------------------------------

    def __getLoopBoundScanningStmts(self, stmts, outer_loop_inames, loop_info_table):
        '''
        Generate an explicit loop-bound scanning code used at runtime to determine the latest start
        and the earliest end of scanning full tiles.
        '''

        # initialize all returned variables
        scan_stmts = []
        lbound_info_seq = []
        new_int_vars = []

        # iterate over each statement
        min_int = ast.NumLitExp(-2147483648, ast.NumLitExp.INT)
        max_int = ast.NumLitExp(2147483647, ast.NumLitExp.INT)
        scan_loops = SimpleLoops()
        for stmt in stmts:
            if not isinstance(stmt, ast.ForStmt):
                lbound_info_seq.append(None)
                continue
            id, lb_exp, ub_exp, st_exp, lbody = self.ast_util.getForLoopInfo(stmt)
            lb_inames = filter(lambda i: self.ast_util.containIdentName(lb_exp, i), outer_loop_inames)
            ub_inames = filter(lambda i: self.ast_util.containIdentName(ub_exp, i), outer_loop_inames)
            if not lb_inames and not ub_inames:
                lbound_info_seq.append(None)
                continue
            lb_name, ub_name = self.__getLoopBoundNames()
            new_int_vars.extend([lb_name, ub_name])
            need_prolog, need_epilog = (len(lb_inames) > 0, len(ub_inames) > 0)
            lbinfo = (lb_name, ub_name, need_prolog, need_epilog)
            lbound_info_seq.append(lbinfo)
            if need_prolog:
                a = ast.BinOpExp(ast.IdentExp(lb_name), min_int.replicate(), ast.BinOpExp.EQ_ASGN)
                scan_stmts.append(ast.ExpStmt(a))
                a = ast.BinOpExp(ast.IdentExp(lb_name),
                                 ast.FunCallExp(ast.IdentExp('max'), [ast.IdentExp(lb_name),
                                                                      lb_exp.replicate()]),
                                 ast.BinOpExp.EQ_ASGN)
                scan_loops.insertLoop(lb_inames, ast.ExpStmt(a))
            else:
                a = ast.BinOpExp(ast.IdentExp(lb_name), lb_exp.replicate(), ast.BinOpExp.EQ_ASGN)
                scan_stmts.append(ast.ExpStmt(a))
            if need_epilog:
                a = ast.BinOpExp(ast.IdentExp(ub_name), max_int.replicate(), ast.BinOpExp.EQ_ASGN)
                scan_stmts.append(ast.ExpStmt(a))
                a = ast.BinOpExp(ast.IdentExp(ub_name),
                                 ast.FunCallExp(ast.IdentExp('min'), [ast.IdentExp(ub_name),
                                                                      ub_exp.replicate()]),
                                 ast.BinOpExp.EQ_ASGN)                
                scan_loops.insertLoop(ub_inames, ast.ExpStmt(a))
            else:
                a = ast.BinOpExp(ast.IdentExp(ub_name), ub_exp.replicate(), ast.BinOpExp.EQ_ASGN)
                scan_stmts.append(ast.ExpStmt(a))
        n_loop_info_table = {}
        for iname, linfo in loop_info_table.items():
            _,_,_,st_exp,_ = linfo
            n_loop_info_table[iname] = (self.__getTileSizeName(iname, self.num_level),
                                        ast.IdentExp(self.__getIterName(iname, self.num_level)),
                                        st_exp)
        scan_stmts.extend(scan_loops.convertToASTs(n_loop_info_table))
                
        # return all necessary information
        return (scan_stmts, lbound_info_seq, new_int_vars)

    #----------------------------------------------

    def __convertToASTs(self, dstmts, loop_inames, loop_info_table):
        '''
        To convert the given list of "modified/decorated" statements to a sequence of AST statements.
        Below is a grammar that describes the given list of "modified/decorated" statements:
          <dstmts> ::= (<bool> , <stmts>) <dstmts>*
                     | <test-exp> <dstmts> <dstmts> <dstmts>*
        '''

        asts = []
        i = 0
        while i < len(dstmts):
            s = dstmts[i]
            if isinstance(s, tuple):
                is_tiled, stmts = s
                stmts = [s.replicate() for s in stmts]
                if is_tiled:
                    asts.extend(stmts)
                else:
                    lbody = ast.CompStmt(stmts)
                    st_exps = []
                    for iname in loop_inames:
                        _,_,_,st_exp,_ = loop_info_table[iname]
                        st_exps.append(st_exp)
                    l = self.__getMultiLevelTileLoop(self.num_level, loop_inames, st_exps, lbody)
                    asts.append(l)
                i += 1
            else:
                if not isinstance(s, ast.BinOpExp):
                    print 'internal error:Tiling: a test expression is expected'
                    sys.exit(1)
                i += 1
                t1 = self.__convertToASTs(dstmts[i], loop_inames, loop_info_table)
                i += 1
                t2 = self.__convertToASTs(dstmts[i], loop_inames, loop_info_table)
                test_exp = s.replicate()
                true_stmt = ast.CompStmt(t1)
                false_stmt = ast.CompStmt(t2)
                asts.append(ast.IfStmt(test_exp, true_stmt, false_stmt))
                i += 1
        return asts

    #----------------------------------------------

    def __tile(self, stmt, new_int_vars, outer_loop_infos, preceding_stmts, lbound_info):
        '''Apply tiling on the given statement'''

        if isinstance(stmt, ast.ExpStmt):
            preceding_stmts = preceding_stmts[:]
            if preceding_stmts:
                is_tiled, last_stmts = preceding_stmts.pop()
                if is_tiled:
                    preceding_stmts.append((is_tiled, last_stmts))                    
                    preceding_stmts.append((False, [stmt]))
                else:
                    preceding_stmts.append((False, last_stmts + [stmt]))
            else:
                preceding_stmts.append((False, [stmt]))
            return preceding_stmts

        elif isinstance(stmt, ast.CompStmt):
            print ('internal error:Tiling: unexpected compound statement directly nested inside ' +
                   'another compound statement')
            sys.exit(1)

        elif isinstance(stmt, ast.IfStmt):
            preceding_stmts = preceding_stmts[:]
            if preceding_stmts:
                is_tiled, last_stmts = preceding_stmts.pop()
                if is_tiled:
                    preceding_stmts.append((is_tiled, last_stmts))                    
                    preceding_stmts.append((False, [stmt]))
                else:
                    preceding_stmts.append((False, last_stmts + [stmt]))
            else:
                preceding_stmts.append((False, [stmt]))
            return preceding_stmts

        elif isinstance(stmt, ast.ForStmt):

            # extract loop structure information
            this_linfo = self.ast_util.getForLoopInfo(stmt) 
            this_id, this_lb_exp, this_ub_exp, this_st_exp, this_lbody = this_linfo
            this_iname = this_id.name
            
            # provide information the (extended) tiled outer loops
            n_outer_loop_infos = outer_loop_infos + [(this_iname, this_linfo)]
            outer_loop_inames = [i for i,_ in outer_loop_infos]
            loop_info_table = dict(outer_loop_infos)
            n_outer_loop_inames = [i for i,_ in n_outer_loop_infos]
            n_loop_info_table = dict(n_outer_loop_infos)

            # get explicit loop-bound scanning code
            t = self.__getLoopBoundScanningStmts(this_lbody.stmts, n_outer_loop_inames,
                                                 n_loop_info_table)
            scan_stmts, lbound_info_seq, ivars = t

            # update the newly declared integer variables
            new_int_vars.extend(ivars)
            
            # prepare loop bounds information
            need_prolog = False
            need_epilog = False     
            rect_lb_exp = this_lb_exp
            rect_ub_exp = this_ub_exp
            if lbound_info:
                lb_name, ub_name, need_prolog, need_epilog = lbound_info
                rect_lb_exp = ast.IdentExp(lb_name)
                rect_ub_exp = ast.IdentExp(ub_name)

            # initialize the resulting statements
            res_stmts = preceding_stmts[:]

            # generate the prolog code
            prolog_code = None
            if need_prolog:
                ub = ast.BinOpExp(rect_lb_exp, ast.ParenthExp(this_st_exp), ast.BinOpExp.SUB)
                prolog_code = self.ast_util.createForLoop(this_id, this_lb_exp, ub,
                                                          this_st_exp, this_lbody)
                if res_stmts:
                    is_tiled, last_stmts = res_stmts.pop()
                    if is_tiled:
                        res_stmts.append((is_tiled, last_stmts))
                        res_stmts.append((False, [prolog_code]))
                    else:
                        res_stmts.append((False, last_stmts + [prolog_code]))
                else:
                    res_stmts.append((False, [prolog_code]))
                
            # generate the main rectangularly tiled code
            processed_stmts = []
            if_branches = [processed_stmts]
            for s, binfo in zip(this_lbody.stmts, lbound_info_seq):
                n_if_branches = []
                for p_stmts in if_branches:
                    if binfo == None:
                        n_p_stmts = self.__tile(s, new_int_vars, n_outer_loop_infos, p_stmts, binfo)
                        while len(p_stmts) > 0:
                            p_stmts.pop()
                        p_stmts.extend(n_p_stmts)
                        n_if_branches.append(p_stmts)
                    else:
                        if len(p_stmts) > 0:
                            p = p_stmts.pop()
                            last_p_stmts = [p]
                        else:
                            last_p_stmts = []
                        n_p_stmts = self.__tile(s, new_int_vars, n_outer_loop_infos,
                                                last_p_stmts, binfo)
                        true_p_stmts = n_p_stmts
                        false_p_stmts = last_p_stmts
                        if false_p_stmts:
                            is_tiled, last_stmts = false_p_stmts.pop()
                            if is_tiled:
                                false_p_stmts.append((is_tiled, last_stmts))                    
                                false_p_stmts.append((False, [s]))
                            else:
                                false_p_stmts.append((False, last_stmts + [s]))
                        else:
                            false_p_stmts.append((False, [s]))
                        lbn,ubn,_,_ = binfo
                        test_exp = ast.BinOpExp(ast.IdentExp(lbn), ast.IdentExp(ubn), ast.BinOpExp.LT)
                        p_stmts.append(test_exp)
                        p_stmts.append(true_p_stmts)
                        p_stmts.append(false_p_stmts)
                        n_if_branches.append(true_p_stmts)
                        n_if_branches.append(false_p_stmts)
                if_branches = n_if_branches
            lbody_stmts = []
            lbody_stmts.extend(scan_stmts)
            lbody_stmts.extend(self.__convertToASTs(processed_stmts, n_outer_loop_inames,
                                                    n_loop_info_table))
            lbody = ast.CompStmt(lbody_stmts)
            iname = self.__getIterName(this_iname, self.num_level)
            tname = self.__getTileSizeName(this_iname, self.num_level)
            main_tiled_code = self.__getInterTileLoop(iname, tname, rect_lb_exp, rect_ub_exp,
                                                      this_st_exp, lbody)
            res_stmts.append((True, [main_tiled_code]))
            
            # generate the cleanup code (the epilog code is already fused with this cleanup code)
            lb = ast.IdentExp(self.__getIterName(this_iname, self.num_level))
            cleanup_code = self.ast_util.createForLoop(this_id, lb, this_ub_exp,
                                                       this_st_exp, this_lbody)
            res_stmts.append((False, [cleanup_code]))

            # return the resulting statements
            return res_stmts

        else:
            print 'internal error:Tiling: unknown type of statement: %s' % stmt.__class__.__name__
            sys.exit(1)

    #----------------------------------------------

    def __startTiling(self, stmt, new_int_vars):
        '''Find the loops to be tiled and then apply loop-tiling transformation on each of them'''

        if isinstance(stmt, ast.ExpStmt):
            return stmt

        elif isinstance(stmt, ast.CompStmt):
            tstmts = []
            for s in stmt.stmts:
                ts = self.__startTiling(s, new_int_vars)
                if isinstance(ts, ast.CompStmt):
                    tstmts.extend(ts.stmts)
                else:
                    tstmts.append(ts)
            stmt.stmts = tstmts
            return stmt

        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__startTiling(stmt.true_stmt, new_int_vars)
            if stmt.false_stmt:
                stmt.false_stmt = self.__startTiling(stmt.false_stmt, new_int_vars)
            return stmt

        elif isinstance(stmt, ast.ForStmt):
            tstmts = []
            for _,stmts in self.__tile(stmt, new_int_vars, [], [], None):
                tstmts.extend(stmts)
            return ast.CompStmt(tstmts)

        else:
            print 'internal error:Tiling: unknown type of statement: %s' % stmt.__class__.__name__
            sys.exit(1)

    #----------------------------------------------

    def transform(self, stmts):
        '''To apply loop-tiling transformation on the given statements'''

        # reset the counter (used for generating new variable names)
        self.counter = 1

        # perform loop tiling on each statement
        new_int_vars = []
        stmts = [self.__startTiling(s, new_int_vars) for s in stmts]

        # return the tiled statements and the newly declared integer variables
        return (stmts, new_int_vars)

