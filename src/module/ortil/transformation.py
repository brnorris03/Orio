#
# The implementation of the code transformation that performs loop tiling
#

import sys
import ast, ast_util
from orio.main.util.globals import *

#-------------------------------------------------

class Transformation:
    '''The code transformation that performs loop tiling'''

    def __init__(self, tiling_info):
        '''To instantiate a code transformation'''

        # unpack the tiling information
        num_tile_level, iter_names = tiling_info

        # the used tiling information
        self.num_tile_level = num_tile_level
        self.iter_names = iter_names

        # a library that provides common AST utility functions
        self.ast_util = ast_util.ASTUtil()

        # used for variable name generation
        self.counter = 1

        # booleans to switch on/off some (not all) of the complementary optimizations
        self.affine_lbound_exps = True         # to use static loop-bound value determination
        self.simplify_one_time_loop = True     # to use one-time loop simplification
        self.use_boundary_tiling = True        # to recursively tile all boundary tiles

        # transformation parameters
        self.recursive_tile_level = self.num_tile_level   # the highest level of tiling used to 
                                                          # further tile the boundary tiles
        #self.recursive_tile_level = 0

    #----------------------------------------------

    def __getTileIterName(self, iter_name, level):
        '''Generate a new variable name for inter-tile loop iterator'''
        return iter_name + ('t%s' % level)

    def __getTileSizeName(self, iter_name, level):
        '''Generate a new variable name for tile size variable'''
        return ('T%s' % level) + iter_name

    def __getLoopBoundNames(self):
        '''Generate new variable names for both lower and upper loop bounds'''
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

    def __getMultiLevelTileLoop(self, tile_level, iter_names, st_exps, lbody):
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
        for level in range(1, tile_level+1):
            if level == 1:
                for iname, st_exp in zip(iter_names, st_exps):
                    n_tsize_name = self.__getTileSizeName(iname, level)
                    lb = ast.IdentExp(self.__getTileIterName(iname, level))
                    loop = self.__getIntraTileLoop(iname, n_tsize_name, lb, st_exp, loop)
            else:
                for iname in iter_names:
                    c_iname = self.__getTileIterName(iname, level-1)
                    n_tsize_name = self.__getTileSizeName(iname, level)
                    lb = ast.IdentExp(self.__getTileIterName(iname, level))
                    st = ast.IdentExp(self.__getTileSizeName(iname, level-1))
                    loop = self.__getIntraTileLoop(c_iname, n_tsize_name, lb, st, loop)
        return loop
    
    #----------------------------------------------

    def __findMinMaxVal(self, min_or_max, exp, var_names, val_table, up_sign = 1):
        '''
        To statically find the actual min/max value of the given expression, based on the given 
        bound variables. The given table records the lowest and highest values of each bound variable.
        The up_sign argument carries the positive/negative sign from the upper level of the AST.
        '''

        # numerical expression
        if isinstance(exp, ast.NumLitExp):
            return exp
        
        # string expression
        elif isinstance(exp, ast.StringLitExp):
            err('orio.module.ortil.transformation: OrTil: invalid string expression found in loop bound expression: %s' % exp)
        
        # identifier expression
        elif isinstance(exp, ast.IdentExp):
            
            # do nothing if the identifier is not in the given list of variables to be replaced
            if exp.name not in var_names:
                return exp
            
            # replace the identifier with its apropriate value (depending on min/max, and upper sign)
            lval, uval = val_table[exp.name]
            if min_or_max == 'max':
                if up_sign == 1:
                    val = ast.ParenthExp(uval.replicate())
                else:
                    val = ast.ParenthExp(lval.replicate())
            elif min_or_max == 'min':
                if up_sign == 1:
                    val = ast.ParenthExp(lval.replicate())
                else:
                    val = ast.ParenthExp(uval.replicate())
            else:
                err('orio.module.ortil.transformation internal error: unrecognized min/max argument value')

            # return the obtained min/max value
            return val

        # array reference expression
        elif isinstance(exp, ast.ArrayRefExp):
            err('orio.module.ortil.transformation: invalid array-reference expression found in loop bound ' +
                   'expression: %s' % exp)

        # function call expression
        elif isinstance(exp, ast.FunCallExp):
            
            # check the function name
            if (not isinstance(exp.exp, ast.IdentExp)) or exp.exp.name not in ('min', 'max'):
                err(('orio.module.ortil.transformation: function name found in loop bound expression must be ' +
                        'min/max, obtained: %s') % exp.exp)

            # recursion on each function argument
            exp.args = []
            for a in exp.args:
                exp.args.append(self.__findMinMaxVal(min_or_max, a, var_names, val_table, up_sign)) 
                

            # return the computed expression
            return exp

        # unary operation expression
        elif isinstance(exp, ast.UnaryExp):
            
            # check the unary operation
            if exp.op_type not in (ast.UnaryExp.PLUS, ast.UnaryExp.MINUS):
                err(('orio.module.ortil.transformation: unary operation found in loop bound expression must ' +
                        'be +/-, obtained: %s') % exp.exp)

            # update the sign, and do recursion on the inner expression
            if exp.op_type == ast.UnaryExp.MINUS:
                up_sign *= -1
            exp.exp = self.__findMinMaxVal(min_or_max, exp.exp, var_names, val_table, up_sign)

            # return the computed expression
            return exp

        # binary operation expression
        elif isinstance(exp, ast.BinOpExp):
            
            # check the binary operation
            if exp.op_type not in (ast.BinOpExp.ADD, ast.BinOpExp.SUB, ast.BinOpExp.MUL):
                err(('orio.module.ortil.transformation: binary operation found in loop bound expression must ' +
                        'be +/-/*, obtained: %s') % exp)

            # do recursion on both operands
            exp.lhs = self.__findMinMaxVal(min_or_max, exp.lhs, var_names, val_table, up_sign)
            if exp.op_type == ast.BinOpExp.SUB:
                up_sign *= -1
            exp.rhs = self.__findMinMaxVal(min_or_max, exp.rhs, var_names, val_table, up_sign)

            # return the computed expression
            return exp

        # parenthesized expression
        elif isinstance(exp, ast.ParenthExp):
            parenth_before = isinstance(exp.exp, ast.ParenthExp)
            exp.exp = self.__findMinMaxVal(min_or_max, exp.exp, var_names, val_table, up_sign)
            parenth_after = isinstance(exp.exp, ast.ParenthExp)
            if (not parenth_before) and parenth_after:
                return exp.exp
            return exp

        # unrecognized expression
        else:
            err('orio.module.ortil.transformation internal error: unknown type of expression: %s' % exp.__class__.__name__)

    #----------------------------------------------

    def __staticLoopBoundScanning(self, stmts, tile_level, outer_loop_inames, loop_info_table):
        ''' 
        Assuming that the loop-bound expressions are affine functions of outer loop iterators and 
        global parameters, we can determine the loop bounds of full tiles in compile time.
        This is an optimization strategy to produce more efficient code.
        Assumptions: 
          1. Lower bound expression must be in the form of: max(e_1,e_2,e_3,...,e_n)
          2. Upper bound expression must be in the form of: min(e_1,e_2,e_3,...,e_n)
          where e_i is an affine function of outer loop iterators and global parameters
        Note that max(x,y,z) is implemented as nested binary max functions: max(z,max(y,z)). The same
        condition applies for min function.
        When n=1, max/min function is not needed.
        '''

        # initialize all returned variables
        scan_stmts = []
        lbound_info_seq = []
        int_vars = []

        # generate the lower and upper values of each inter-tile loop
        val_table = {}
        for iname in outer_loop_inames:
            _,_,_,st_exp,_ = loop_info_table[iname]
            lval = ast.IdentExp(self.__getTileIterName(iname, tile_level))
            t = ast.BinOpExp(ast.IdentExp(self.__getTileSizeName(iname, tile_level)),
                             ast.ParenthExp(st_exp.replicate()), ast.BinOpExp.SUB)
            uval = ast.BinOpExp(lval.replicate(), ast.ParenthExp(t), ast.BinOpExp.ADD)
            val_table[iname] = (lval, uval)

        # iterate over each statement to determine loop bounds that are affine functions 
        # of outer loop iterators
        lb_exps_table = {}
        ub_exps_table = {}
        for stmt in stmts:
            
            # skip all non loop statements
            if not isinstance(stmt, ast.ForStmt):
                lbound_info_seq.append(None)
                continue

            # extract this loop structure
            id, lb_exp, ub_exp, st_exp, lbody = self.ast_util.getForLoopInfo(stmt)

            # see if the loop bound expressions are bound/free of outer loop iterators 
            lb_inames = filter(lambda i: self.ast_util.containIdentName(lb_exp, i), outer_loop_inames)
            ub_inames = filter(lambda i: self.ast_util.containIdentName(ub_exp, i), outer_loop_inames)

            # skip loops with bound expressions that are free of outer loop iterators
            if not lb_inames and not ub_inames:
                lbound_info_seq.append(None)
                continue

            # check if this loop runs only once
            is_one_time_loop = str(lb_exp) == str(ub_exp)

            # generate booleans to indicate the needs of prolog, epilog, and orio.main.tiled loop
            if is_one_time_loop:
                need_tiled_loop = False
                need_prolog = False
                need_epilog = False
            else:
                need_tiled_loop = True
                need_prolog = len(lb_inames) > 0
                need_epilog = len(ub_inames) > 0

            # generate new variable names for both the new lower and upper loop bounds
            if need_tiled_loop:
                lb_name, ub_name = self.__getLoopBoundNames()
                int_vars.extend([lb_name, ub_name])
            else:
                lb_name = ''
                ub_name = ''

            # append information about the new loop bounds
            lbinfo = (lb_name, ub_name, need_prolog, need_epilog, need_tiled_loop)
            lbound_info_seq.append(lbinfo)

            # skip generating loop-bound scanning code (if it's a one-time loop)
            if not need_tiled_loop:
                continue

            # determine the value of the new lower loop bound
            if str(lb_exp) in lb_exps_table:
                lb_var = lb_exps_table[str(lb_exp)]
                a = ast.BinOpExp(ast.IdentExp(lb_name), lb_var.replicate(), ast.BinOpExp.EQ_ASGN)
            else:
                if need_prolog:
                    t = self.__findMinMaxVal('max', lb_exp.replicate(), lb_inames, val_table)
                    a = ast.BinOpExp(ast.IdentExp(lb_name), t.replicate(), ast.BinOpExp.EQ_ASGN)
                else:
                    a = ast.BinOpExp(ast.IdentExp(lb_name), lb_exp.replicate(), ast.BinOpExp.EQ_ASGN)
                lb_exps_table[str(lb_exp)] = ast.IdentExp(lb_name)
            scan_stmts.append(ast.ExpStmt(a))

            # determine the value of the new upper loop bound
            if str(ub_exp) in ub_exps_table:
                ub_var = ub_exps_table[str(ub_exp)]
                a = ast.BinOpExp(ast.IdentExp(ub_name), ub_var.replicate(), ast.BinOpExp.EQ_ASGN)
            else:
                if need_epilog:
                    t = self.__findMinMaxVal('min', ub_exp.replicate(), ub_inames, val_table)
                    a = ast.BinOpExp(ast.IdentExp(ub_name), t.replicate(), ast.BinOpExp.EQ_ASGN)
                else:
                    a = ast.BinOpExp(ast.IdentExp(ub_name), ub_exp.replicate(), ast.BinOpExp.EQ_ASGN)
                ub_exps_table[str(ub_exp)] = ast.IdentExp(ub_name)
            scan_stmts.append(ast.ExpStmt(a))
        
        # return all necessary information
        return (scan_stmts, lbound_info_seq, int_vars)

    #----------------------------------------------

    def __getLoopBoundScanningStmts(self, stmts, tile_level, outer_loop_inames, loop_info_table):
        '''
        Generate an explicit loop-bound scanning code used at runtime to determine the latest start
        and the earliest end of scanning full tiles.
        '''

        # (optimization) generate code that determines the loop bounds of full tiles at compile time
        if self.affine_lbound_exps:
            return self.__staticLoopBoundScanning(stmts, tile_level, outer_loop_inames,
                                                  loop_info_table)

        # initialize all returned variables
        scan_stmts = []
        lbound_info_seq = []
        int_vars = []

        # iterate over each statement to find loop bounds that are functions of outer loop iterators
        min_int = ast.NumLitExp(-2147483648, ast.NumLitExp.INT)
        max_int = ast.NumLitExp(2147483647, ast.NumLitExp.INT)
        lb_exps_table = {}
        ub_exps_table = {}
        pre_scan_stmts = []
        post_scan_stmts = []
        scan_loops = SimpleLoops()
        for stmt in stmts:
        
            # skip all non loop statements
            if not isinstance(stmt, ast.ForStmt):
                lbound_info_seq.append(None)
                continue

            # extract this loop structure
            id, lb_exp, ub_exp, st_exp, lbody = self.ast_util.getForLoopInfo(stmt)

            # see if the loop bound expressions are bound/free of outer loop iterators 
            lb_inames = filter(lambda i: self.ast_util.containIdentName(lb_exp, i), outer_loop_inames)
            ub_inames = filter(lambda i: self.ast_util.containIdentName(ub_exp, i), outer_loop_inames)

            # skip loops with bound expressions that are free of outer loop iterators
            if not lb_inames and not ub_inames:
                lbound_info_seq.append(None)
                continue

            # check if this loop runs only once
            is_one_time_loop = str(lb_exp) == str(ub_exp)

            # generate booleans to indicate the needs of prolog, epilog, and orio.main.tiled loop
            if is_one_time_loop:
                need_tiled_loop = False
                need_prolog = False
                need_epilog = False
            else:
                need_tiled_loop = True
                need_prolog = len(lb_inames) > 0
                need_epilog = len(ub_inames) > 0

            # generate new variable names for both the new lower and upper loop bounds
            if need_tiled_loop:
                lb_name, ub_name = self.__getLoopBoundNames()
                int_vars.extend([lb_name, ub_name])
            else:
                lb_name = ''
                ub_name = ''

            # append information about the new loop bounds
            lbinfo = (lb_name, ub_name, need_prolog, need_epilog, need_tiled_loop)
            lbound_info_seq.append(lbinfo)

            # skip generating loop-bound scanning code (if it's a one-time loop)
            if not need_tiled_loop:
                continue

            # generate loop-bound scanning code for the prolog
            if str(lb_exp) in lb_exps_table:
                lb_var = lb_exps_table[str(lb_exp)]
                a = ast.BinOpExp(ast.IdentExp(lb_name), lb_var.replicate(), ast.BinOpExp.EQ_ASGN)
                post_scan_stmts.append(ast.ExpStmt(a))
            else:
                if need_prolog:
                    a = ast.BinOpExp(ast.IdentExp(lb_name), min_int.replicate(), ast.BinOpExp.EQ_ASGN)
                    pre_scan_stmts.append(ast.ExpStmt(a))
                    a = ast.BinOpExp(ast.IdentExp(lb_name),
                                     ast.FunCallExp(ast.IdentExp('max'), [ast.IdentExp(lb_name),
                                                                          lb_exp.replicate()]),
                                     ast.BinOpExp.EQ_ASGN)
                    scan_loops.insertLoop(lb_inames, ast.ExpStmt(a))
                else:
                    a = ast.BinOpExp(ast.IdentExp(lb_name), lb_exp.replicate(), ast.BinOpExp.EQ_ASGN)
                    pre_scan_stmts.append(ast.ExpStmt(a))
                lb_exps_table[str(lb_exp)] = ast.IdentExp(lb_name)

            # generate loop-bound scaning code for the epilog
            if str(ub_exp) in ub_exps_table:
                ub_var = ub_exps_table[str(ub_exp)]
                a = ast.BinOpExp(ast.IdentExp(ub_name), ub_var.replicate(), ast.BinOpExp.EQ_ASGN)
                post_scan_stmts.append(ast.ExpStmt(a))
            else:
                if need_epilog:
                    a = ast.BinOpExp(ast.IdentExp(ub_name), max_int.replicate(), ast.BinOpExp.EQ_ASGN)
                    pre_scan_stmts.append(ast.ExpStmt(a))
                    a = ast.BinOpExp(ast.IdentExp(ub_name),
                                     ast.FunCallExp(ast.IdentExp('min'), [ast.IdentExp(ub_name),
                                                                          ub_exp.replicate()]),
                                     ast.BinOpExp.EQ_ASGN)                
                    scan_loops.insertLoop(ub_inames, ast.ExpStmt(a))
                else:
                    a = ast.BinOpExp(ast.IdentExp(ub_name), ub_exp.replicate(), ast.BinOpExp.EQ_ASGN)
                    pre_scan_stmts.append(ast.ExpStmt(a))
                ub_exps_table[str(ub_exp)] = ast.IdentExp(ub_name)

        # build a new loop information tabe for generating the loop-bound scanning code
        n_loop_info_table = {}
        for iname, linfo in loop_info_table.items():
            _,_,_,st_exp,_ = linfo
            n_loop_info_table[iname] = (self.__getTileSizeName(iname, tile_level),
                                        self.__getTileIterName(iname, tile_level), st_exp)

        # convert the "SimpleLoop" abstractions into loop ASTs
        scan_loop_stmts = scan_loops.convertToASTs(tile_level, n_loop_info_table)

        # merge all scanning statements
        scan_stmts = pre_scan_stmts + scan_loop_stmts + post_scan_stmts
                
        # return all necessary information
        return (scan_stmts, lbound_info_seq, int_vars)

    #----------------------------------------------

    def __convertToASTs(self, dstmts, tile_level, contain_loop, loop_inames, loop_info_table,
                        int_vars):
        '''
        To recursively convert the given list, containing processed statements and possibly 
        if-branching statements, to AST. A sample of the given list is as follows:
           [s1, t-exp, [s2, s3], [s4]]
        which represents the following AST:
           s1; if (t-exp) {s2; s3;} else s4;
        '''

        # initialize the list of ASTs
        asts = []

        # iterating over each list element
        i = 0
        while i < len(dstmts):
            s = dstmts[i]

            # if it's an AST
            if isinstance(s, tuple):
                is_tiled, stmts = s
                stmts = [s.replicate() for s in stmts]

                # already tiled; no need to enclose with an inter-tile loop nest
                if is_tiled:
                    asts.extend(stmts)

                # need to enclose with a single-level inter-tile loop nest
                elif contain_loop:

                    # add single-level tiled loop
                    rev_inames = loop_inames[:]
                    rev_inames.reverse()
                    l = ast.CompStmt(stmts)
                    for iname in rev_inames:
                        tname = self.__getTileSizeName(iname, tile_level)
                        lb_exp = ast.IdentExp(self.__getTileIterName(iname, tile_level))
                        _,_,_,st_exp,_ = loop_info_table[iname]
                        lbody = l             
                        l = self.__getIntraTileLoop(iname, tname, lb_exp, st_exp, lbody)
                        l.fully_tiled = True

                    # to recursively tile all boundary tiles
                    if self.use_boundary_tiling:
                        new_tile_level = min(tile_level-1, self.recursive_tile_level)
                        if new_tile_level > 0:
                            l = self.__startTiling(l, new_tile_level, int_vars)

                    # insert the tiled loop to the AST list
                    asts.append(l)

                # need to enclose with a multilevel inter-tile loop nest
                else:
                    lbody = ast.CompStmt(stmts)
                    st_exps = []
                    for iname in loop_inames:
                        _,_,_,st_exp,_ = loop_info_table[iname]
                        st_exps.append(st_exp)
                    l = self.__getMultiLevelTileLoop(tile_level, loop_inames, st_exps, lbody)
                    asts.append(l)                    

                # increment index
                i += 1

            # if it's an if-statement's test expression
            else:
                if not isinstance(s, ast.BinOpExp):
                    err('orio.module.ortil.transformation internal error: a test expression is expected')

                # generate AST for the true statement
                t1 = self.__convertToASTs(dstmts[i+1], tile_level, contain_loop,
                                          loop_inames, loop_info_table, int_vars)

                # generate AST for the false statement
                t2 = self.__convertToASTs(dstmts[i+2], tile_level, contain_loop,
                                          loop_inames, loop_info_table, int_vars)

                # generate AST for the if-statement
                test_exp = s.replicate()
                true_stmt = ast.CompStmt(t1)
                false_stmt = ast.CompStmt(t2)
                asts.append(ast.IfStmt(test_exp, true_stmt, false_stmt))

                # increment index
                i += 3

        # return the list of ASTs
        return asts

    #----------------------------------------------

    def __labelFullCoreTiledLoop(self, stmt, outer_loop_inames):
        '''To traverse the given statement and label the fully rectangularly tiled loop nest'''

        # there is no outer tiled loops
        if not outer_loop_inames:
            return

        # find the starting loop nest that iterates the full rectangular tiles
        s = stmt
        while True:
            if not isinstance(s, ast.ForStmt):
                break
            if s.start_label and s.start_label.startswith('start full core tiles: '):
                break 
            id, lb_exp, ub_exp, st_exp, lbody = self.ast_util.getForLoopInfo(s)
            if id.name == outer_loop_inames[0]:
                s.start_label = 'start full core tiles: '
                s.end_label = 'end full core tiles: '
                for i,iname in enumerate(outer_loop_inames):
                    if i:
                        s.start_label += ','
                        s.end_label += ','
                    s.start_label += '%s' % iname
                    s.end_label += '%s' % iname
                break
            if len(s.stmt.stmts) != 1:
                break
            s = s.stmt.stmts[0]

    #----------------------------------------------

    def __tile(self, stmt, tile_level, outer_loop_infos, preceding_stmts, lbound_info, int_vars):
        '''Apply tiling on the given statement'''

        # complain if the tiling level is not a positive integer
        if not isinstance(tile_level, int) or tile_level <= 0:
            err('orio.module.ortil.transformation internal error: invalid tiling level: %s' % tile_level)

        # cannot handle a directly nested compound statement
        if isinstance(stmt, ast.CompStmt):
           err('orio.module.ortil.transformation internal error: unexpected compound statement directly nested inside ' +
                   'another compound statement')

        # to handle the given expression statement or if statement
        if isinstance(stmt, ast.ExpStmt) or isinstance(stmt, ast.IfStmt):
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

        # to tile the given for-loop statement
        if isinstance(stmt, ast.ForStmt):

            # check if this loop is already tiled and whether it's fully tiled
            this_fully_tiled = stmt.fully_tiled
            
            # extract loop structure information
            this_linfo = self.ast_util.getForLoopInfo(stmt) 
            this_id, this_lb_exp, this_ub_exp, this_st_exp, this_lbody = this_linfo
            this_iname = this_id.name
            
            # get information about the (extended) tiled outer loops
            outer_loop_inames = [i for i,_ in outer_loop_infos]
            loop_info_table = dict(outer_loop_infos)
            n_outer_loop_infos = outer_loop_infos + [(this_iname, this_linfo)]
            n_outer_loop_inames = [i for i,_ in n_outer_loop_infos]
            n_loop_info_table = dict(n_outer_loop_infos)

            # prepare loop bounds information (for iterating full rectangular tiles)
            need_prolog = False
            need_epilog = False
            rect_lb_exp = this_lb_exp
            rect_ub_exp = this_ub_exp
            if lbound_info:
                lb_name, ub_name, need_prolog, need_epilog, need_tiled_loop = lbound_info
                rect_lb_exp = ast.IdentExp(lb_name)
                rect_ub_exp = ast.IdentExp(ub_name)
                if not need_tiled_loop:
                    err('orio.module.ortil.transformation internal error: unexpected case where generation of the orio.main.' +
                           'rectangular tiled loop is needed')

            # get explicit loop-bound scanning code
            t = self.__getLoopBoundScanningStmts(this_lbody.stmts, tile_level, n_outer_loop_inames,
                                                 n_loop_info_table)
            scan_stmts, lbound_info_seq, ivars = t

            # update the newly declared integer variables
            int_vars.extend(ivars)
            
            # initialize the resulting statements
            res_stmts = preceding_stmts[:]

            # generate the prolog code
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
                
            # start generating the orio.main.rectangularly tiled code 
            # (note: the body of the tiled code may contain if-statement branches, 
            # each needed to be recursively transformed)
            # example of the resulting processed statements:
            #   s1; if (t-exp) {s2; s3;} else s4;
            # is represented as the following list:
            #   [s1, t-exp, [s2, s3], [s4]]
            contain_loop = False
            processed_stmts = []
            if_branches = [processed_stmts]     # a container for storing list of if-branch statements
            for s, binfo in zip(this_lbody.stmts, lbound_info_seq):

                # check if one of the enclosed statements is a loop
                if isinstance(s, ast.ForStmt):
                    contain_loop = True

                # perform transformation on each if-branch statements
                n_if_branches = []
                for p_stmts in if_branches:

                    # replicate the statement (just to be safe)
                    s = s.replicate()

                    # this is NOT a loop statement with bound expressions that are functions of
                    # outer loop iterators
                    if binfo == None:
                        n_p_stmts = self.__tile(s, tile_level, n_outer_loop_infos,
                                                p_stmts, binfo, int_vars)
                        while len(p_stmts) > 0:
                            p_stmts.pop()
                        p_stmts.extend(n_p_stmts)
                        n_if_branches.append(p_stmts)
                        continue

                    # (optimization) special handling for one-time loop --> remove the if's true
                    # condition (i.e., lb<ub) since it will never be executed.
                    _,_,_,_,need_tiled_loop = binfo
                    if not need_tiled_loop:
                        if p_stmts:
                            is_tiled, last_stmts = p_stmts.pop()
                            if is_tiled:
                                p_stmts.append((is_tiled, last_stmts))
                                p_stmts.append((False, [s]))
                            else:
                                p_stmts.append((False, last_stmts + [s]))
                        else:
                            p_stmts.append((False, [s]))
                        n_if_branches.append(p_stmts)
                        continue

                    # (optimization) recursively feed in the last processed statement only, and 
                    # leave the other preceeding statements untouched --> for reducing code size
                    if len(p_stmts) > 0:
                        p = p_stmts.pop()
                        last_p_stmts = [p]
                    else:
                        last_p_stmts = []

                    # perform a recursion to further tile this rectangularly tiled loop
                    n_p_stmts = self.__tile(s.replicate(), tile_level, n_outer_loop_infos,
                                            last_p_stmts, binfo, int_vars)

                    # compute the processed statements for both true and false conditions
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

                    # create two sets of if-branch statements
                    lbn,ubn,_,_,_ = binfo
                    test_exp = ast.BinOpExp(ast.IdentExp(lbn), ast.IdentExp(ubn), ast.BinOpExp.LT)
                    p_stmts.append(test_exp)
                    p_stmts.append(true_p_stmts)
                    p_stmts.append(false_p_stmts)
                    n_if_branches.append(true_p_stmts)
                    n_if_branches.append(false_p_stmts)

                # update the if-branch statements
                if_branches = n_if_branches

            # combine the loop-bound scanning statements
            lbody_stmts = []
            lbody_stmts.extend(scan_stmts)
            
            # convert the processed statements into AST
            lbody_stmts.extend(self.__convertToASTs(processed_stmts, tile_level, contain_loop,
                                                    n_outer_loop_inames, n_loop_info_table, int_vars))

            # generate the orio.main.rectangularly tiled code
            lbody = ast.CompStmt(lbody_stmts)
            iname = self.__getTileIterName(this_iname, tile_level)
            tname = self.__getTileSizeName(this_iname, tile_level)
            tiled_code = self.__getInterTileLoop(iname, tname, rect_lb_exp, rect_ub_exp,
                                                      this_st_exp, lbody)
            res_stmts.append((True, [tiled_code]))
            
            # mark the loop if it's a loop iterating the full rectangular tiles
            self.__labelFullCoreTiledLoop(tiled_code, n_outer_loop_inames)
            
            # generate the cleanup code (the epilog is already fused)
            if not this_fully_tiled:
                lb = ast.IdentExp(self.__getTileIterName(this_iname, tile_level))
                cleanup_code = self.ast_util.createForLoop(this_id, lb, this_ub_exp,
                                                           this_st_exp, this_lbody)
                res_stmts.append((False, [cleanup_code]))

            # return the resulting statements
            return res_stmts
        
        # unknown statement
        err('orio.module.ortil.transformation internal error: unknown type of statement: %s' % stmt.__class__.__name__)

    #----------------------------------------------

    def __startTiling(self, stmt, tile_level, int_vars):
        '''To apply loop-tiling transformation on each of loop'''
        
        # expression statement
        if isinstance(stmt, ast.ExpStmt):
            return stmt

        # compound statement
        elif isinstance(stmt, ast.CompStmt):
            tstmts = []
            for s in stmt.stmts:
                ts = self.__startTiling(s, tile_level, int_vars)
                if isinstance(ts, ast.CompStmt):
                    tstmts.extend(ts.stmts)
                else:
                    tstmts.append(ts)
            stmt.stmts = tstmts
            return stmt

        # if statement
        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__startTiling(stmt.true_stmt, tile_level, int_vars)
            if stmt.false_stmt:
                stmt.false_stmt = self.__startTiling(stmt.false_stmt, tile_level, int_vars)
            return stmt

        # for loop statement
        elif isinstance(stmt, ast.ForStmt):

            # apply loop tiling on this loop
            tiling_results = self.__tile(stmt, tile_level, [], [], None, int_vars)

            # return the tiled AST
            t_stmts = []
            for is_tiled, stmts in tiling_results:
                if self.use_boundary_tiling and not is_tiled:
                    new_tile_level = min(tile_level-1, self.recursive_tile_level)
                    if new_tile_level > 0:
                        stmts = [self.__startTiling(s, new_tile_level, int_vars) for s in stmts]
                t_stmts.extend(stmts)
            tiled_ast = ast.CompStmt(t_stmts)

            # return the tiled AST
            return tiled_ast

        # unknown statement
        else:
            err('orio.module.ortil.transformation internal error: unknown type of statement: %s' % stmt.__class__.__name__)

    #----------------------------------------------

    def __simplifyOneTimeLoop(self, stmt):
        '''Simplify all one-time loops. This is safe; guaranteed by CLooG.'''
        
        # expression statement
        if isinstance(stmt, ast.ExpStmt):
            return stmt

        # compound statement
        elif isinstance(stmt, ast.CompStmt):
            tstmts = []
            for s in stmt.stmts:
                ts = self.__simplifyOneTimeLoop(s)
                if isinstance(ts, ast.CompStmt):
                    tstmts.extend(ts.stmts)
                else:
                    tstmts.append(ts)
            stmt.stmts = tstmts
            return stmt

        # if statement
        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__simplifyOneTimeLoop(stmt.true_stmt)
            if stmt.false_stmt:
                stmt.false_stmt = self.__simplifyOneTimeLoop(stmt.false_stmt)
            return stmt

        # for loop statement
        elif isinstance(stmt, ast.ForStmt):

            # recurse on the loop body
            id, lb_exp, ub_exp, st_exp, lbody = self.ast_util.getForLoopInfo(stmt) 
            tbody = self.__simplifyOneTimeLoop(stmt.stmt)
            
            # check if it's one-time loop
            if str(lb_exp) == str(ub_exp):
                return tbody
            else:
                stmt.stmt = tbody
                return stmt

        # unknown statement
        else:
            err('orio.module.ortil.transformation internal error: unknown type of statement: %s' % stmt.__class__.__name__)

    #----------------------------------------------

    def __removeInvalidFullCoreLoops(self, stmt):
        '''To remove invalid full-core loops that may appear after one-time loop removal'''

        # expression statement
        if isinstance(stmt, ast.ExpStmt):
            return

        # compound statement
        elif isinstance(stmt, ast.CompStmt):
            for s in stmt.stmts:
                self.__removeInvalidFullCoreLoops(s)

        # if statement
        elif isinstance(stmt, ast.IfStmt):
            self.__removeInvalidFullCoreLoops(stmt.true_stmt)
            if stmt.false_stmt:
                self.__removeInvalidFullCoreLoops(stmt.false_stmt)

        # for loop statement
        elif isinstance(stmt, ast.ForStmt):
            self.__removeInvalidFullCoreLoops(stmt.stmt)

            # check if this is a labeled full core tile loop
            if stmt.start_label.startswith('start full core tiles: '):

                # check the validity of the full core tile loop
                ipos = stmt.start_label.find(':')
                iter_names = stmt.start_label[ipos+1:].split(',')
                iter_names = [i.strip() for i in iter_names]
                valid = True
                cur_stmt = stmt
                while iter_names:
                    if not isinstance(cur_stmt, ast.ForStmt):
                        valid = False
                        break
                    id,_,_,_,_ = self.ast_util.getForLoopInfo(cur_stmt)
                    if id.name != iter_names[0]:
                        valid = False
                        break
                    iter_names.pop(0)
                    if not iter_names:
                        if isinstance(cur_stmt.stmt, ast.CompStmt):
                            stmts = cur_stmt.stmt.stmts[:]
                            contain_loop = False
                            while stmts:
                                s = stmts.pop(0)
                                if isinstance(s, ast.ForStmt):
                                    contain_loop = True
                                    break
                                if isinstance(s, ast.CompStmt):
                                    stmts.extend(s.stmts)
                                elif isinstance(s, ast.IfStmt):
                                    stmts.append(s.true_stmt)
                                    if s.false_stmt:
                                        stmts.append(s.false_stmt)
                            if contain_loop:
                                valid = False
                                break
                        else:
                            if isinstance(cur_stmt.stmt, ast.ForStmt):
                                valid = False
                                break
                        break
                    if isinstance(cur_stmt.stmt, ast.CompStmt):
                        if len(cur_stmt.stmt.stmts) != 1:
                            valid = False
                            break
                        cur_stmt = cur_stmt.stmt.stmts[0]
                    else:
                        cur_stmt = cur_stmt.stmt

                # remove labels if it's not a valid full core tile loop
                if not valid:
                    stmt.start_label = ''
                    stmt.end_label = ''

        # unknown statement
        else:
            err('orio.module.ortil.transformation internal error: unknown type of statement: %s' % stmt.__class__.__name__)

    #----------------------------------------------

    def transform(self, stmts):
        '''To apply loop-tiling transformation on the given statements'''

        # reset the counter (used for generating new variable names)
        self.counter = 1

        # add new integer variable names used for both inter-tile and intra-tile loop iterators
        int_vars = []
        int_vars.extend(self.iter_names)
        for iname in self.iter_names:
            for level in range(1, self.num_tile_level+1):
                int_vars.append(self.__getTileIterName(iname, level))

        # perform loop tiling on each statement
        stmts = [self.__startTiling(s, self.num_tile_level, int_vars) for s in stmts]

        # perform replication of AST (just to be safe -- to avoid a node sharing the AST structure)
        stmts = [s.replicate() for s in stmts]

        # (optimization) simplify one-time loop
        if self.simplify_one_time_loop:
            stmts = [self.__simplifyOneTimeLoop(s) for s in stmts]

        # remove all invalid labeled full core tiles loops that may appear after 
        # removing all one-time loops
        for s in stmts:
            self.__removeInvalidFullCoreLoops(s)

        # return the tiled statements and the newly declared integer variables
        return (stmts, int_vars)

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

        # fusing is not possible, thus create a new loop and add it into the sequence of loops 
        if not is_done:
            inames_seq.reverse()
            l = stmt
            for iname in inames_seq:
                l = (iname, [l])
            loops.append(l)

    #----------------------------------------------

    def convertToASTs(self, tile_level, loop_info_table, node = None):
        '''To generate a sequence of ASTs that correspond to this sequence of "simple" loops'''

        if node == None:
            node = self.loops
        
        if isinstance(node, tuple):
            iname, subnodes = node
            tsize_name, titer_name, st_exp = loop_info_table[iname]
            id = ast.IdentExp(iname)
            lb = ast.IdentExp(titer_name)
            tmp = ast.BinOpExp(ast.IdentExp(tsize_name), ast.ParenthExp(st_exp), ast.BinOpExp.SUB)
            ub = ast.BinOpExp(lb, ast.ParenthExp(tmp), ast.BinOpExp.ADD)
            st = st_exp
            lbody = ast.CompStmt(self.convertToASTs(tile_level, loop_info_table, subnodes))
            return self.ast_util.createForLoop(id, lb, ub, st, lbody)

        elif isinstance(node, list):
            return [self.convertToASTs(tile_level, loop_info_table, n) for n in node]

        else:
            return node


