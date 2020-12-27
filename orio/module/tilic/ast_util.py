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
        '''To instantiate the AST utility'''
        pass

    #---------------------------------------------------------------

    def getLoopIters(self, stmt):
        '''Return all loop iterator names used in the given statement'''

        if isinstance(stmt, ast.ExpStmt):
            return []
        
        elif isinstance(stmt, ast.CompStmt):
            inames = []
            for s in stmt.stmts:
                for i in self.getLoopIters(s):
                    if i not in inames:
                        inames.append(i)
            return inames

        elif isinstance(stmt, ast.IfStmt):
            inames = self.getLoopIters(stmt.true_stmt)
            if stmt.false_stmt:
                for i in self.getLoopIters(stmt.false_stmt):
                    if i not in inames:
                        inames.append(i)
            return inames
                    
        elif isinstance(stmt, ast.ForStmt):
            id,_,_,_,_ = self.getForLoopInfo(stmt)
            inames = [id.name]
            for i in self.getLoopIters(stmt.stmt):
                if i not in inames:
                    inames.append(i)
            return inames
                
        else:
            err('orio.module.tilic.ast_util internal error: unknown type of statement: %s' % stmt.__class__.__name__)
            
    #---------------------------------------------------------------

    def containIdentName(self, exp, iname):
        '''Check if the given expression contains an identifier whose name matches with the given name'''

        if not isinstance(exp, ast.Exp):
            err('orio.module.tilic.ast_util internal error: input not an expression type')

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
            err('orio.module.tilic.ast_util  internal error: unexpected AST type: "%s"' % exp.__class__.__name__)
            
    #---------------------------------------------------------------

    def getForLoopInfo(self, stmt):
        '''
        Return information about the loop structure.
        for (id = lb; id <= ub; id = id + st)
          bod
        '''

        # get rid of compound statement that contains only a single statement
        while isinstance(stmt, ast.CompStmt) and len(stmt.stmts) == 1:
            stmt = stmt.stmts[0]

        # check if it is a loop
        if not isinstance(stmt, ast.ForStmt):
            err('orio.module.tilic.ast_util: Tilic: input not a loop statement')

        # check loop initialization
        if stmt.init:
            if not (isinstance(stmt.init, ast.BinOpExp) and stmt.init.op_type == ast.BinOpExp.EQ_ASGN and 
                    isinstance(stmt.init.lhs, ast.IdentExp)):
                err('orio.module.tilic.ast_util: Tilic: loop initialization not in "id = lb" form')
                
        # check loop test
        if stmt.test:
            if not (isinstance(stmt.test, ast.BinOpExp) and stmt.test.op_type in (ast.BinOpExp.LE, ast.BinOpExp.LT) and 
                    isinstance(stmt.test.lhs, ast.IdentExp)):
                err('orio.module.tilic.ast_util: Tilic: loop test not in "id <= ub" or "id < ub" form')

        # check loop iteration
        if stmt.iter:
            if not ((isinstance(stmt.iter, ast.BinOpExp) and stmt.iter.op_type == ast.BinOpExp.EQ_ASGN and 
                     isinstance(stmt.iter.lhs, ast.IdentExp) and isinstance(stmt.iter.rhs, ast.BinOpExp) and
                     isinstance(stmt.iter.rhs.lhs, ast.IdentExp) and stmt.iter.rhs.op_type == ast.BinOpExp.ADD and 
                     stmt.iter.lhs.name == stmt.iter.rhs.lhs.name) 
                    or
                    (isinstance(stmt.iter, ast.UnaryExp) and isinstance(stmt.iter.exp, ast.IdentExp) and
                     stmt.iter.op_type in (ast.UnaryExp.PRE_INC, ast.UnaryExp.POST_INC))):
                err('orio.module.tilic.ast_util: Tilic: loop iteration not in "id++" or "id += st" or "id = id + st" form')

        # check if the control expressions are all empty
        if not stmt.init and not stmt.test and not stmt.iter:
            err('orio.module.tilic.ast_util: Tilic: loop with an empty control expression cannot be handled')
    
        # check if the iterator names in the control expressions are all the same
        inames = []
        if stmt.init:
            inames.append(stmt.init.lhs.name)
        if stmt.test:
            inames.append(stmt.test.lhs.name)
        if stmt.iter:
            if isinstance(stmt.iter, ast.BinOpExp):
                inames.append(stmt.iter.lhs.name)
            else:
                inames.append(stmt.iter.exp.name)
        if inames.count(inames[0]) != len(inames):
            err('orio.module.tilic.ast_util: Tilic: different iterator names used in the loop control expressions')
        
        # extract the loop structure information
        id = ast.IdentExp(inames[0])
        lb = None
        ub = None
        st = None
        if stmt.init:
            lb = stmt.init.rhs.replicate()
        if stmt.test:
            if stmt.test.op_type == ast.BinOpExp.LT:
                ub = ast.BinOpExp(stmt.test.rhs.replicate(), ast.NumLitExp(1, ast.NumLitExp.INT), ast.BinOpExp.SUB)
            else:
                ub = stmt.test.rhs.replicate()
        if stmt.iter:
            if isinstance(stmt.iter, ast.BinOpExp):
                st = stmt.iter.rhs.rhs.replicate()
            else:
                st = ast.NumLitExp(1, ast.NumLitExp.INT)
        bod = stmt.stmt.replicate()
        
        # return the loop structure information
        return (id, lb, ub, st, bod)

    #---------------------------------------------------------------

    def createForLoop(self, id, lb, ub, st, bod):
        '''
        Generate a loop statement:
        for (id = lb; id <= ub; id = id + st)
          bod
        '''

        init = None
        test = None
        iter = None
        if lb:
            init = ast.BinOpExp(id.replicate(), lb.replicate(), ast.BinOpExp.EQ_ASGN)
        if ub:
            test = ast.BinOpExp(id.replicate(), ub.replicate(), ast.BinOpExp.LE)
        if st:
            i = ast.BinOpExp(id.replicate(), st.replicate(), ast.BinOpExp.ADD)
            iter = ast.BinOpExp(id.replicate(), i, ast.BinOpExp.EQ_ASGN)
        bod = bod.replicate()
        if not isinstance(bod, ast.CompStmt):
            bod = ast.CompStmt([bod])
        return ast.ForStmt(init, test, iter, bod)
    

