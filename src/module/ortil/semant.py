#
# The implementation of the semantic analyzer
#

import sys
import ast, ast_util

#---------------------------------------------------------

class SemanticAnalyzer:
    '''The semantic analyzer class that provides methods for check and enforcing AST semantics'''

    def __init__(self, tiling_info):
        '''To instantiate a semantic analyzer'''

        num_level, iter_names = tiling_info

        self.num_level = num_level
        self.iter_names = iter_names
        self.ast_util = ast_util.ASTUtil()

    #-----------------------------------------------------

    def __normalizeStmt(self, stmt):
        '''
        * To change the format of all for-loops to a fixed form as described below:
           for (<id> = <exp>; <id> <= <exp>; <id> += <exp>) {
             <stmts>
           }
        * To change the format of all if-statements to a fixed form as described below:
           if (<exp>) {
             <stmts>
           } else {
             <stmts>
           } 
        * To remove meaningless scopings inside each compound statement.
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
            print 'internal error:OrTil: unknown type of statement: %s' % stmt.__class__.__name__
            sys.exit(1)
                
    #-----------------------------------------------------

    def __checkStmt(self, stmt, oloop_inames = []):
        '''
        * To complain if there is a sublevel of compound statement directly nested inside
          a compound statement.
        * To complain if there is an illegal loop nest, where an inner loop has the same iterator
          name as the outer loop. The following instance is invalid:
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
                    print ('error:OrTil: does not support a compound statement directly nested ' +
                           'inside another compound statement')
                    sys.exit(1)
        
        elif isinstance(stmt, ast.IfStmt):
            self.__checkStmt(stmt.true_stmt, oloop_inames)
            if stmt.false_stmt:
                self.__checkStmt(stmt.false_stmt, oloop_inames)

        elif isinstance(stmt, ast.ForStmt):
            (id, lb, ub, st, bod) = self.ast_util.getForLoopInfo(stmt)
            if id.name in oloop_inames:
                print ('error:OrTil: illegal loop nest where an inner loop has the same iterator ' +
                       'name as the outer loop')
                sys.exit(1)
            if id.name not in self.iter_names:
                print 'error:OrTil: missing tiled-loop iterator name: "%s"' % id.name
                sys.exit(1)
            self.__checkStmt(stmt.stmt, oloop_inames + [id.name])
            
        else:
            print 'internal error:OrTil: unknown type of statement: %s' % stmt.__class__.__name__
            sys.exit(1)

    #-----------------------------------------------------

    def analyze(self, stmts):
        '''To check and enforce AST semantics'''

        # normalize the given statements
        stmts = [self.__normalizeStmt(s) for s in stmts]

        # check the correctness of the AST semantics
        for s in stmts:
            self.__checkStmt(s)

        # check if there is a specified loop iterator that is not used in the actual code
        used_iter_names = {}
        for s in stmts:
            inames = self.ast_util.getLoopIters(s)
            for i in inames:
                used_iter_names[i] = None
        for i in self.iter_names:
            if i not in used_iter_names:
                print 'error:OrTil: unused tiled-loop iterator name: "%s"' % i
                sys.exit(1)

        # return the semantically correct statements
        return stmts
        
        
