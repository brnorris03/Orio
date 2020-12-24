#
# A library for for-loop statements
#

import sets, sys
import orio.module.loop.ast
from orio.main.util.globals import *

#-----------------------------------------

class ForLoopLib:
    '''A library tool used to provide a set of subroutines to process for-loop statements'''

    def __init__(self):
        '''To instantiate a for-loop library tool object'''
        pass

    #-------------------------------------------------

    def extractForLoopInfo(self, stmt):
        '''
        Given a for-loop statement, extract information about its loop structure
        Note that the for-loop must be in the following form:
          for (<id> = <exp>; <id> <= <exp>; <id> += <exp>)
            <stmt>
        Subtraction is not considered at the iteration expression for the sake of
        the implementation simplicity.
        '''
    

        # get rid of compound statement that contains only a single statement
        while isinstance(stmt, orio.module.loop.ast.CompStmt) and len(stmt.stmts) == 1:
            stmt = stmt.stmts[0]

        # check if it is a for-loop statement
        if not isinstance(stmt, orio.module.loop.ast.ForStmt):
            err('orio.module.loop.ast_lib.forloop_lib: %s: not a for-loop statement' % stmt.line_no)

        # check initialization expression
        if stmt.init:
            while True:
                while isinstance(stmt.init, orio.module.loop.ast.ParenthExp):
                    stmt.init = stmt.init.exp
                if (isinstance(stmt.init, orio.module.loop.ast.BinOpExp) and
                    stmt.init.op_type == orio.module.loop.ast.BinOpExp.EQ_ASGN):
                    while isinstance(stmt.init.lhs, orio.module.loop.ast.ParenthExp):
                        stmt.init.lhs = stmt.init.lhs.exp
                    while isinstance(stmt.init.rhs, orio.module.loop.ast.ParenthExp):
                        stmt.init.rhs = stmt.init.rhs.exp
                    if isinstance(stmt.init.lhs, orio.module.loop.ast.IdentExp): 
                        break
                err('orio.module.loop.ast_lib.forloop_lib:%s: loop initialization expression not in "<id> = <exp>" form' %
                       stmt.init.line_no)
                
        # check test expression
        if stmt.test:
            while True:
                while isinstance(stmt.test, orio.module.loop.ast.ParenthExp):
                    stmt.test = stmt.test.exp
                if (isinstance(stmt.test, orio.module.loop.ast.BinOpExp) and
                    #(
                     stmt.test.op_type == orio.module.loop.ast.BinOpExp.LE
                     # TODO: relax this restriction
                     #or stmt.test.op_type == orio.module.loop.ast.BinOpExp.LT)
                    ):
                    while isinstance(stmt.test.lhs, orio.module.loop.ast.ParenthExp):
                        stmt.test.lhs = stmt.test.lhs.exp
                    while isinstance(stmt.test.rhs, orio.module.loop.ast.ParenthExp):
                        stmt.test.rhs = stmt.test.rhs.exp
                    if isinstance(stmt.test.lhs, orio.module.loop.ast.IdentExp): 
                        break
                err('orio.module.loop.ast_lib.forloop_lib:%s: loop test expression not in "<id> <= <exp>" form' %
                       stmt.test.line_no)
            
        # check iteration expression
        if stmt.iter:
            while True:
                while isinstance(stmt.iter, orio.module.loop.ast.ParenthExp):
                    stmt.iter = stmt.iter.exp
                if (isinstance(stmt.iter, orio.module.loop.ast.BinOpExp) and
                    stmt.iter.op_type == orio.module.loop.ast.BinOpExp.EQ_ASGN):
                    while isinstance(stmt.iter.lhs, orio.module.loop.ast.ParenthExp):
                        stmt.iter.lhs = stmt.iter.lhs.exp
                    while isinstance(stmt.iter.rhs, orio.module.loop.ast.ParenthExp):
                        stmt.iter.rhs = stmt.iter.rhs.exp
                    if isinstance(stmt.iter.lhs, orio.module.loop.ast.IdentExp):
                        if (isinstance(stmt.iter.rhs, orio.module.loop.ast.BinOpExp) and
                            stmt.iter.rhs.op_type in (orio.module.loop.ast.BinOpExp.ADD,
                                                      orio.module.loop.ast.BinOpExp.SUB)):
                            while isinstance(stmt.iter.rhs.lhs, orio.module.loop.ast.ParenthExp):
                                stmt.iter.rhs.lhs = stmt.iter.rhs.lhs.exp
                            while isinstance(stmt.iter.rhs.rhs, orio.module.loop.ast.ParenthExp):
                                stmt.iter.rhs.rhs = stmt.iter.rhs.rhs.exp
                            if (isinstance(stmt.iter.rhs.lhs, orio.module.loop.ast.IdentExp) and
                                stmt.iter.lhs.name == stmt.iter.rhs.lhs.name):
                                break
                elif (isinstance(stmt.iter, orio.module.loop.ast.UnaryExp) and
                      stmt.iter.op_type in (orio.module.loop.ast.UnaryExp.POST_INC,
                                            orio.module.loop.ast.UnaryExp.PRE_INC,
                                            orio.module.loop.ast.UnaryExp.POST_DEC,
                                            orio.module.loop.ast.UnaryExp.PRE_DEC)):
                    while isinstance(stmt.iter.exp, orio.module.loop.ast.ParenthExp):
                        stmt.iter.exp = stmt.iter.exp.exp
                    if isinstance(stmt.iter.exp, orio.module.loop.ast.IdentExp):
                        break
                err(('orio.module.loop.ast_lib.forloop_lib:%s: loop iteration expression not in "<id>++" or "<id>--" or ' +
                        '"<id> += <exp>" or "<id> = <id> + <exp>" form') % stmt.iter.line_no)

        # check if the control expressions are all empty
        if not stmt.init and not stmt.test and not stmt.iter:
            err('orio.module.loop.ast_lib.forloop_lib:%s: a loop with an empty control expression cannot be handled' %
                   stmt.line_no)
    
        # check if the iterator names are all the same
        init_iname = None
        test_iname = None
        iter_iname = None
        if stmt.init:
            init_iname = stmt.init.lhs.name
        if stmt.test:
            test_iname = stmt.test.lhs.name
        if stmt.iter:
            if isinstance(stmt.iter, orio.module.loop.ast.BinOpExp):
                iter_iname = stmt.iter.lhs.name
            else:
                assert(isinstance(stmt.iter, orio.module.loop.ast.UnaryExp)), 'internal error: not unary'
                iter_iname = stmt.iter.exp.name
        inames = []
        if init_iname:
            inames.append(init_iname)
        if test_iname:
            inames.append(test_iname)
        if iter_iname:
            inames.append(iter_iname)
        if inames.count(inames[0]) != len(inames):
            err('orio.module.loop.ast_lib.forloop_lib:%s: iterator names across init, test, and iter exps must be the same'
                   % stmt.line_no)
        
        # extract for-loop structure information
        index_id = orio.module.loop.ast.IdentExp(inames[0])
        lbound_exp = None
        ubound_exp = None
        stride_exp = None
        if stmt.init:
            lbound_exp = stmt.init.rhs.replicate()
        if stmt.test:
            ubound_exp = stmt.test.rhs.replicate()
        if stmt.iter:
            if isinstance(stmt.iter, orio.module.loop.ast.BinOpExp):
                stride_exp = stmt.iter.rhs.rhs.replicate()
                if isinstance(stride_exp, orio.module.loop.ast.BinOpExp):
                    stride_exp = orio.module.loop.ast.ParenthExp(stride_exp)
                if stmt.iter.rhs.op_type == orio.module.loop.ast.BinOpExp.SUB:
                    stride_exp = orio.module.loop.ast.UnaryExp(stride_exp, orio.module.loop.ast.UnaryExp.MINUS)
            elif isinstance(stmt.iter, orio.module.loop.ast.UnaryExp):
                if stmt.iter.op_type in (orio.module.loop.ast.UnaryExp.POST_INC,
                                         orio.module.loop.ast.UnaryExp.PRE_INC):
                    stride_exp = orio.module.loop.ast.NumLitExp(1, orio.module.loop.ast.NumLitExp.INT)
                elif stmt.iter.op_type in (orio.module.loop.ast.UnaryExp.POST_DEC,
                                           orio.module.loop.ast.UnaryExp.PRE_DEC):
                    stride_exp = orio.module.loop.ast.NumLitExp(-1, orio.module.loop.ast.NumLitExp.INT)
                else:
                    err('orio.module.loop.ast_lib.forloop_lib internal error: unexpected unary operation type')
            else:
                err('orio.module.loop.ast_lib.forloop_lib internal error: unexpected type of iteration expression')

        loop_body = stmt.stmt.replicate()
        for_loop_info = (index_id, lbound_exp, ubound_exp, stride_exp, loop_body)
        
        # return the for-loop structure information
        debug("forloop_lib: extractForLoopInfo returning", obj=self, level=6)
        return for_loop_info

    #-------------------------------------------------
    
    def createForLoop(self, index_id, lbound_exp, ubound_exp, stride_exp, loop_body, meta=''):
        '''
        Generate a for loop:
          for (index_id = lbound_exp; index_id <= ubound_exp; index_id = index_id + stride_exp)
            loop_body
        '''

        init_exp = None
        test_exp = None
        iter_exp = None
        if lbound_exp:
            init_exp = orio.module.loop.ast.BinOpExp(index_id.replicate(),
                                                lbound_exp.replicate(),
                                                orio.module.loop.ast.BinOpExp.EQ_ASGN)
        if ubound_exp:
            test_exp = orio.module.loop.ast.BinOpExp(index_id.replicate(),
                                                ubound_exp.replicate(),
                                                orio.module.loop.ast.BinOpExp.LE)
        if stride_exp:
            while isinstance(stride_exp, orio.module.loop.ast.ParenthExp):
                stride_exp = stride_exp.exp
            it = orio.module.loop.ast.BinOpExp(index_id.replicate(),
                                          stride_exp.replicate(),
                                          orio.module.loop.ast.BinOpExp.ADD)
            iter_exp = orio.module.loop.ast.BinOpExp(index_id.replicate(),
                                                it,
                                                orio.module.loop.ast.BinOpExp.EQ_ASGN)
        return orio.module.loop.ast.ForStmt(init_exp, test_exp, iter_exp, loop_body.replicate(), meta={'kind':meta})
    
    #-------------------------------------------------

    def getLoopIndexNames(self, stmt):
        '''Return a list of all loop index names'''

        if stmt == None:
            return []
        
        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            return []

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            inames = []
            for s in stmt.stmts:
                inames.extend(self.getLoopIndexNames(s))
            return list(sets.Set(inames))

        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            inames = []
            inames.extend(self.getLoopIndexNames(stmt.true_stmt))
            if stmt.false_stmt:
                inames.extend(self.getLoopIndexNames(stmt.false_stmt))
            return list(sets.Set(inames))

        elif isinstance(stmt, orio.module.loop.ast.ForStmt) and stmt:
            inames = []
            inames.extend(self.getLoopIndexNames(stmt.stmt))
            index_id, lbound_exp, ubound_exp, stride_exp, loop_body = self.extractForLoopInfo(stmt)
            if index_id.name not in inames:
                inames.append(index_id.name)
            return list(sets.Set(inames))

        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.ast_lib.forloop_lib internal error: unprocessed transform statement')
                        
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return []

        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return []

        else:
            err('orio.module.loop.ast_lib.forloop_lib internal error: unexpected AST type: "%s"' % stmt.__class__.__name__)

    #-------------------------------------------------

    def hasInnerLoop(self, stmt):
        '''Determine if there is an inner loop inside the given statement'''

        if stmt == None:
            return False
        
        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            return False

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            for s in stmt.stmts:
                if self.hasInnerLoop(s):
                    return True
            return False

        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            if self.hasInnerLoop(stmt.true_stmt):
                return True
            else:
                return self.hasInnerLoop(stmt.false_stmt)

        elif isinstance(stmt, orio.module.loop.ast.ForStmt):
            return True

        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.ast_lib.forloop_lib internal error: unprocessed transform statement')
        
        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return False
                
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return False

        else:
            err('orio.module.loop.ast_lib.forloop_lib internal error: unexpected AST type: "%s"' % stmt.__class__.__name__)
