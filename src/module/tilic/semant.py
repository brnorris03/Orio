#
# The implementation of the semantic checker of the AST
#

import sys
import ast, ast_util
from orio.main.util.globals import *

#---------------------------------------------------------

class SemanticChecker:
    '''The semantic checker class that provides methods for checking and enforcing the AST semantics'''

    def __init__(self):
        '''To instantiate the semantic checker'''

        self.ast_util = ast_util.ASTUtil()

    #-----------------------------------------------------

    def __normalizeStmt(self, stmt):
        '''
        * To enforce all loops to have a fixed loop form:
           for (id = lb; id <= ub; id += st) {
             bod
           }
        * To enforce all if-statements to have a fixed if-statement form:
           if (test) {
             true-case
           } else {
             false-case
           }
        * To remove all meaningless scopings inside each compound statement.
        '''

        if isinstance(stmt, ast.ExpStmt):
            return stmt

        elif isinstance(stmt, ast.CompStmt):
            stmt.stmts = [self.__normalizeStmt(s) for s in stmt.stmts]
            while len(stmt.stmts) == 1 and isinstance(stmt.stmts[0], ast.CompStmt):
                stmt.stmts = stmt.stmts[0].stmts
            return stmt
        
        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__normalizeStmt(stmt.true_stmt)
            if not isinstance(stmt.true_stmt, ast.CompStmt):
                stmt.true_stmt = ast.CompStmt([stmt.true_stmt])
            if stmt.false_stmt:
                stmt.false_stmt = self.__normalizeStmt(stmt.false_stmt)
                if not isinstance(stmt.false_stmt, ast.CompStmt):
                    stmt.false_stmt = ast.CompStmt([stmt.false_stmt])
            return stmt

        elif isinstance(stmt, ast.ForStmt):
            stmt.stmt = self.__normalizeStmt(stmt.stmt)
            (id, lb, ub, st, bod) = self.ast_util.getForLoopInfo(stmt)
            if not isinstance(bod, ast.CompStmt):
                bod = ast.CompStmt([bod])
            stmt = self.ast_util.createForLoop(id, lb, ub, st, bod)
            return stmt
        
        else:
            err('orio.module.tilic.semant internal error: unknown type of statement: %s' % stmt.__class__.__name__)
                
    #-----------------------------------------------------

    def __checkStmt(self, stmt, oloop_inames = []):
        '''
        * To complain if there is a compound statement that is directly nested in another compound statement.
        * To complain if an inner loop uses the same iterator name as the outer loop.
          for i
           for i
            S
        '''

        if isinstance(stmt, ast.ExpStmt):
            pass

        elif isinstance(stmt, ast.CompStmt):
            for s in stmt.stmts:
                self.__checkStmt(s, oloop_inames)
            for s in stmt.stmts:
                if isinstance(s, ast.CompStmt):
                    err('orio.module.tilic.semant: Tilic: a compound statement cannot be directly nested in another compound statement')
        
        elif isinstance(stmt, ast.IfStmt):
            self.__checkStmt(stmt.true_stmt, oloop_inames)
            if stmt.false_stmt:
                self.__checkStmt(stmt.false_stmt, oloop_inames)

        elif isinstance(stmt, ast.ForStmt):
            (id, lb, ub, st, bod) = self.ast_util.getForLoopInfo(stmt)
            if id.name in oloop_inames:
                err('orio.module.tilic.semant: Tilic: illegal loop nest where an inner loop has the same iterator name as the outer loop')
            self.__checkStmt(stmt.stmt, oloop_inames + [id.name])
            
        else:
            err('orio.module.tilic.semant internal error: unknown type of statement: %s' % stmt.__class__.__name__)

    #-----------------------------------------------------

    def check(self, stmts):
        '''To check and enforce the AST semantics'''

        # normalize the given statements
        stmts = [self.__normalizeStmt(s) for s in stmts]

        # check the correctness of the AST semantics
        for s in stmts:
            self.__checkStmt(s)

        # return the semantically correct statements
        return stmts
        
        
