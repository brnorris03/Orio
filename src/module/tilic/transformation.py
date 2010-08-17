#
# The implementation of the code transformation that performs loop tiling
#

import sys
from orio.main.util.globals import *
import ast, ast_util

#-------------------------------------------------

class Transformation:
    '''The code transformation that performs loop tiling'''

    def __init__(self, tiling_params):
        '''To instantiate the code transformation'''

        # unpack the tiling parameters
        (num_tiling_levels, first_depth, last_depth, 
         max_boundary_tiling_level, affine_lbound_exps) = tiling_params

        # set the tiling parameters
        self.num_tiling_levels = num_tiling_levels
        self.first_depth = first_depth
        self.last_depth = last_depth
        self.max_boundary_tiling_level = max_boundary_tiling_level
        self.affine_lbound_exps = affine_lbound_exps

        # a library that provides common AST utility functions
        self.ast_util = ast_util.ASTUtil()

        # used for generating new variable names
        self.counter = 1

    #----------------------------------------------

    def __tile(self, stmt, loop_depth, tile_level, ):
        '''To apply tiling transformation on the given statement'''

        return ([], [stmt])

    #----------------------------------------------

    def __startTiling(self, stmt):
        '''To apply tiling transformation on the top-level loop statement'''

        # expression statement
        if isinstance(stmt, ast.ExpStmt):
            return ([], stmt)

        # compound statement
        elif isinstance(stmt, ast.CompStmt):
            int_vars = []
            tstmts = []
            for s in stmt.stmts:
                ivars, ts = self.__startTiling(s)
                int_vars.extend(ivars)
                if isinstance(ts, ast.CompStmt):
                    tstmts.extend(ts.stmts)
                else:
                    tstmts.append(ts)
            stmt.stmts = tstmts
            return (int_vars, stmt)

        # if statement
        elif isinstance(stmt, ast.IfStmt):
            int_vars = []
            ivars, ts = self.__startTiling(stmt.true_stmt)
            int_vars.extend(ivars)
            stmt.true_stmt = ts
            if stmt.false_stmt:
                ivars, ts = self.__startTiling(stmt.false_stmt)
                int_vars.extend(ivars)                
                stmt.false_stmt = ts
            return (int_vars, stmt)

        # loop statement
        elif isinstance(stmt, ast.ForStmt):
            int_vars, tstmts = self.__tile(stmt)
            if len(tstmts) > 1:
                tiled_stmt = ast.CompStmt(tstmts)
            else:
                tiled_stmt = tstmts[0]
            return (int_vars, tiled_stmt)

        # unknown statement
        else:
            err('orio.module.tilic.transformation internal error: unknown type of statement: %s' % stmt.__class__.__name__)

    #----------------------------------------------

    def __removeOneTimeLoops(self, stmt):
        '''Remove all one-time loops. This is safe (guaranteed by CLooG).'''

        # expression statement
        if isinstance(stmt, ast.ExpStmt):
            return stmt

        # compound statement
        elif isinstance(stmt, ast.CompStmt):
            tstmts = []
            for s in stmt.stmts:
                ts = self.__removeOneTimeLoops(s)
                if isinstance(ts, ast.CompStmt):
                    tstmts.extend(ts.stmts)
                else:
                    tstmts.append(ts)
            stmt.stmts = tstmts
            return stmt

        # if statement
        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__removeOneTimeLoops(stmt.true_stmt)
            if stmt.false_stmt:
                stmt.false_stmt = self.__removeOneTimeLoops(stmt.false_stmt)
            return stmt

        # loop statement
        elif isinstance(stmt, ast.ForStmt):
            id, lb, ub, st, bod = self.ast_util.getForLoopInfo(stmt)
            tbod = self.__removeOneTimeLoops(bod)
            if str(lb) == str(ub):
                return tbod
            else:
                stmt.stmt = tbod
                return stmt

        # unknown statement
        else:
            err('orio.module.tilic.transformation internal error: unknown type of statement: %s' % stmt.__class__.__name__)

    #----------------------------------------------

    def transform(self, stmts):
        '''To apply tiling transformation on the given statements'''

        # reset the counter
        self.counter = 1

        # perform the tiling transformation
        int_vars = []
        tstmts = []
        for s in stmts:
            ivars, ts = self.__startTiling(s)
            for i in ivars:
                if i not in int_vars:
                    int_vars.append(i)
            tstmts.append(ts)
        stmts = tstmts

        # perform replications of ASTs (just to be safe)
        stmts = [s.replicate() for s in stmts]

        # remove all one-time loops
        stmts = [self.__removeOneTimeLoops(s) for s in stmts]

        # return the tiled statements and the newly declared integer variables
        return (stmts, int_vars)

