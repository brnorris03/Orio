#
# Provides utility methods for AST traversal and manipulation
#

import sys
from . import ast
from orio.main.util.globals import *
from functools import reduce

#-------------------------------------------------------------------

class ASTUtil:
    '''A class definition that provides utility methods for AST traversal and manipulation'''

    def __init__(self):
        '''To instantiate an AST utility'''
        pass

    #---------------------------------------------------------------

    def containIdentName(self, exp, iname):
        '''
        Check if the given expression contains an identifier whose name matches to the given name
        '''

        if not isinstance(exp, ast.Exp):
            err('orio.module.ortildriver.ast_util internal error:containIdentName: input not an expression type')

        if exp == None:
            return False
        
        if isinstance(exp, ast.NumLitExp):
            return False
        
        elif isinstance(exp, ast.StringLitExp):
            return False
        
        elif isinstance(exp, ast.IdentExp):
            return exp.name == iname
        
        elif isinstance(exp, ast.ArrayRefExp):
            return self.containIdentName(exp.exp, iname) or self.containIdentName(exp.sub_exp, iname)
        
        elif isinstance(exp, ast.FunCallExp):
            has_match = reduce(lambda x,y: x or y,
                               [self.containIdentName(a, iname) for a in exp.args],
                               False)
            return self.containIdentName(exp.exp, iname) or has_match
        
        elif isinstance(exp, ast.UnaryExp):
            return self.containIdentName(exp.exp, iname)
        
        elif isinstance(exp, ast.BinOpExp):
            return self.containIdentName(exp.lhs, iname) or self.containIdentName(exp.rhs, iname)
        
        elif isinstance(exp, ast.ParenthExp):
            return self.containIdentName(exp.exp, iname)
        
        else:
            err('orio.module.ortildriver.ast_util internal error: unexpected AST type: "%s"' % exp.__class__.__name__)
            
    #---------------------------------------------------------------

    def getForLoopInfo(self, stmt):
        '''
        Given a for-loop statement, extract information about its loop structure.
        Note that the for-loop must be in the following form:
          for (<id> = <exp>; <id> <= <exp>; <id> += <exp>)
            <stmt>
        Subtraction is not considered at the iteration expression for the sake of
        the implementation simplicity.
        '''

        # get rid of compound statement that contains only a single statement
        while isinstance(stmt, ast.CompStmt) and len(stmt.stmts) == 1:
            stmt = stmt.stmts[0]

        # check if it is a for-loop statement
        if not isinstance(stmt, ast.ForStmt):
            err('orio.module.ortildriver.ast_util: OrTilDriver:%s: not a for-loop statement' % stmt.line_no)

        # check initialization expression
        if stmt.init:
            while True:
                while isinstance(stmt.init, ast.ParenthExp):
                    stmt.init = stmt.init.exp
                if (isinstance(stmt.init, ast.BinOpExp) and
                    stmt.init.op_type == ast.BinOpExp.EQ_ASGN):
                    while isinstance(stmt.init.lhs, ast.ParenthExp):
                        stmt.init.lhs = stmt.init.lhs.exp
                    while isinstance(stmt.init.rhs, ast.ParenthExp):
                        stmt.init.rhs = stmt.init.rhs.exp
                    if isinstance(stmt.init.lhs, ast.IdentExp): 
                        break
                err('orio.module.ortildriver.ast_util:%s: loop initialization expression not in "<id> = <exp>" form' %
                       stmt.init.line_no)
                
        # check test expression
        if stmt.test:
            while True:
                while isinstance(stmt.test, ast.ParenthExp):
                    stmt.test = stmt.test.exp
                if (isinstance(stmt.test, ast.BinOpExp) and
                    stmt.test.op_type in (ast.BinOpExp.LT, ast.BinOpExp.LE)):
                    while isinstance(stmt.test.lhs, ast.ParenthExp):
                        stmt.test.lhs = stmt.test.lhs.exp
                    while isinstance(stmt.test.rhs, ast.ParenthExp):
                        stmt.test.rhs = stmt.test.rhs.exp
                    if isinstance(stmt.test.lhs, ast.IdentExp): 
                        break
                err('orio.module.ortildriver.ast_util:%s: loop test expression not in "<id> <= <exp>" or ' +
                       '"<id> < <exp>"form' % stmt.test.line_no)
            
        # check iteration expression
        if stmt.iter:
            while True:
                while isinstance(stmt.iter, ast.ParenthExp):
                    stmt.iter = stmt.iter.exp
                if (isinstance(stmt.iter, ast.BinOpExp) and
                    stmt.iter.op_type == ast.BinOpExp.EQ_ASGN):
                    while isinstance(stmt.iter.lhs, ast.ParenthExp):
                        stmt.iter.lhs = stmt.iter.lhs.exp
                    while isinstance(stmt.iter.rhs, ast.ParenthExp):
                        stmt.iter.rhs = stmt.iter.rhs.exp
                    if isinstance(stmt.iter.lhs, ast.IdentExp):
                        if (isinstance(stmt.iter.rhs, ast.BinOpExp) and
                            stmt.iter.rhs.op_type in (ast.BinOpExp.ADD, ast.BinOpExp.SUB)):
                            while isinstance(stmt.iter.rhs.lhs, ast.ParenthExp):
                                stmt.iter.rhs.lhs = stmt.iter.rhs.lhs.exp
                            while isinstance(stmt.iter.rhs.rhs, ast.ParenthExp):
                                stmt.iter.rhs.rhs = stmt.iter.rhs.rhs.exp
                            if (isinstance(stmt.iter.rhs.lhs, ast.IdentExp) and
                                stmt.iter.lhs.name == stmt.iter.rhs.lhs.name):
                                break
                elif (isinstance(stmt.iter, ast.UnaryExp) and
                      stmt.iter.op_type in (ast.UnaryExp.POST_INC, ast.UnaryExp.PRE_INC,
                                            ast.UnaryExp.POST_DEC, ast.UnaryExp.PRE_DEC)):
                    while isinstance(stmt.iter.exp, ast.ParenthExp):
                        stmt.iter.exp = stmt.iter.exp.exp
                    if isinstance(stmt.iter.exp, ast.IdentExp):
                        break
                err(('orio.module.ortildriver.ast_util:%s: loop iteration expression not in "<id>++" or "<id>--" or ' +
                        '"<id> += <exp>" or "<id> = <id> + <exp>" form') % stmt.iter.line_no)

        # check if the control expressions are all empty
        if not stmt.init and not stmt.test and not stmt.iter:
            err('orio.module.ortildriver.ast_util:%s: a loop with an empty control expression cannot be handled' %
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
            if isinstance(stmt.iter, ast.BinOpExp):
                iter_iname = stmt.iter.lhs.name
            else:
                assert(isinstance(stmt.iter, ast.UnaryExp)), 'internal error:OrTilDriver: not unary'
                iter_iname = stmt.iter.exp.name
        inames = []
        if init_iname:
            inames.append(init_iname)
        if test_iname:
            inames.append(test_iname)
        if iter_iname:
            inames.append(iter_iname)
        if inames.count(inames[0]) != len(inames):
            err('orio.module.ortildriver.ast_util:%s: iterator names across init, test, and iter exps must be the same'
                   % stmt.line_no)
        
        # extract for-loop structure information
        index_id = ast.IdentExp(inames[0])
        lbound_exp = None
        ubound_exp = None
        stride_exp = None
        if stmt.init:
            lbound_exp = stmt.init.rhs.replicate()
        if stmt.test:
            if stmt.test.op_type == ast.BinOpExp.LT:
                ubound_exp = ast.BinOpExp(stmt.test.rhs.replicate(),
                                          ast.NumLitExp(1, ast.NumLitExp.INT), ast.BinOpExp.SUB)
            else:
                ubound_exp = stmt.test.rhs.replicate()
        if stmt.iter:
            if isinstance(stmt.iter, ast.BinOpExp):
                stride_exp = stmt.iter.rhs.rhs.replicate()
                if isinstance(stride_exp, ast.BinOpExp):
                    stride_exp = ast.ParenthExp(stride_exp)
                if stmt.iter.rhs.op_type == ast.BinOpExp.SUB:
                    stride_exp = ast.UnaryExp(stride_exp, ast.UnaryExp.MINUS)
            elif isinstance(stmt.iter, ast.UnaryExp):
                if stmt.iter.op_type in (ast.UnaryExp.POST_INC, ast.UnaryExp.PRE_INC):
                    stride_exp = ast.NumLitExp(1, ast.NumLitExp.INT)
                elif stmt.iter.op_type in (ast.UnaryExp.POST_DEC, ast.UnaryExp.PRE_DEC):
                    stride_exp = ast.NumLitExp(-1, ast.NumLitExp.INT)
                else:
                    err('orio.module.ortildriver.ast_util internal error: unexpected unary operation type')
            else:
                err('orio.module.ortildriver.ast_util internal error: unexpected type of iteration expression')
        loop_body = stmt.stmt.replicate()
        for_loop_info = (index_id, lbound_exp, ubound_exp, stride_exp, loop_body)
        
        # return the for-loop structure information
        return for_loop_info

    #---------------------------------------------------------------

    def createForLoop(self, index_id, lbound_exp, ubound_exp, stride_exp, loop_body):
        '''
        Generate a for-loop statement based on the given loop structure information:
          for (index_id = lbound_exp; index_id <= ubound_exp; index_id = index_id + stride_exp)
            loop_body
        '''

        init_exp = None
        test_exp = None
        iter_exp = None
        if lbound_exp:
            init_exp = ast.BinOpExp(index_id.replicate(), lbound_exp.replicate(),
                                    ast.BinOpExp.EQ_ASGN)
        if ubound_exp:
            test_exp = ast.BinOpExp(index_id.replicate(), ubound_exp.replicate(), ast.BinOpExp.LE)
        if stride_exp:
            while isinstance(stride_exp, ast.ParenthExp):
                stride_exp = stride_exp.exp
            it = ast.BinOpExp(index_id.replicate(), stride_exp.replicate(), ast.BinOpExp.ADD)
            iter_exp = ast.BinOpExp(index_id.replicate(), it, ast.BinOpExp.EQ_ASGN)
        if not isinstance(loop_body, ast.CompStmt):
            loop_body = ast.CompStmt([loop_body])
        return ast.ForStmt(init_exp, test_exp, iter_exp, loop_body.replicate())
    
