#
# The implementation of the code transformator that performs loop tiling
#

import sys
import ast, ast_util

#-------------------------------------------------

class Transformator:
    '''The code transformator that performs loop tiling'''

    def __init__(self, perf_params, tiling_info):
        '''To instantiate a code transformator'''

        self.perf_params = perf_params
        self.num_level, self.tiling_table = tiling_info
        self.ast_util = ast_util.ASTUtil()

    #----------------------------------------------

    def __getNewIteratorName(self, index_name, level):
        '''Generate a new iterator name for inter-tile loop'''
        return index_name + ('t' * level)

    def __getNewTileSizeName(self, index_name, level):
        '''Generate a new variable name for the tile size'''
        return 'T' + (index_name * level)

    #----------------------------------------------

    def __createInterTileLoop(self, level, index_name, lbound_exp, ubound_exp, stride_exp, loop_body):
        '''
        Generate an inter-tile loop
          for (it=lb; it<=ub-(Ti-St); it+=Ti)
            <loop-body>
        '''
        
        id = ast.IdentExp(self.__getNewIteratorName(index_name, level))
        st = ast.IdentExp(self.__getNewTileSizeName(index_name, level))
        lb = lbound_exp.replicate()
        tmp = ast.BinOpExp(self.__getNewTileSizeName(index_name, level),
                           ast.ParenthExp(stride_exp.replicate()), ast.BinOpExp.SUB)
        ub = ast.BinOpExp(ubound_exp.replicate(), ast.ParenthExp(tmp), ast.BinOpExp.SUB)
        bod = loop_body.replicate()
        return self.ast_util.createForLoop(id, lb, ub, st, bod)

    #----------------------------------------------

    def __createIntraTileLoop(self, level, index_name, stride_exp, loop_body):
        '''
        Generate an intra-tile loop:
          for (i=it; i<=it+(Ti-St); i+=St)
            <loop-body>
        '''
        
        id = ast.IdentExp(index_name)
        st = stride_exp.replicate()
        lb = ast.IdentExp(self.__getNewIteratorName(index_name, level))
        tmp = ast.BinOpExp(self.__getNewTileSizeName(index_name, level),
                           ast.ParenthExp(stride_exp.replicate()), ast.BinOpExp.SUB)
        ub = ast.BinOpExp(ast.IdentExp(self.__getNewIteratorName(index_name, level)),
                          ast.ParenthExp(tmp), ast.BinOpExp.ADD)
        bod = loop_body.replicate()        
        return self.ast_util.createForLoop(id, lb, ub, st, bod)

    #----------------------------------------------

    def __tile(self, stmt, new_integer_vars, outer_loops, preceding_untiled_stmts):
        '''Apply tiling on the given statement'''

        if isinstance(stmt, ast.ExpStmt):
            tiled_stmts = []
            untiled_stmts = preceding_untiled_stmts + [stmt]
            return (tiled_stmts, untiled_stmts)

        elif isinstance(stmt, ast.CompStmt):
            print ('internal error:Tiling: unexpected compound statement directly nested inside ' +
                   'another compound statement')
            sys.exit(1)

        elif isinstance(stmt, ast.IfStmt):
            return ([], preceding_untiled_stmts + [stmt])

        elif isinstance(stmt, ast.ForStmt):

            # first get all needed information
            for_loop_info = self.ast_util.getForLoopInfo(stmt)
            (index_id, lbound_exp, ubound_exp, stride_exp, loop_body) = for_loop_info
            outer_loop_inames = [iname for iname, linfo in outer_loops]
            outer_loop_infos = [linfo for iname, linfo in outer_loops]

            # check if this loop does not have any tiling information
            if index_id.name not in self.tiling_table:
                print ('error:Tiling: missing loop "%s" in the list of loops to be tiled' %
                       index_id.name)
                sys.exit(1)

            # find all outer loop iterator names that are used in the loop expressions
            bound_inames = []
            for i in outer_loop_inames:
                if (self.ast_util.containIdentName(lbound_exp, index_id.name) or
                    self.ast_util.containIdentName(ubound_exp, index_id.name)):
                    bound_inames.append(i)

            #  
                
                
                
            
            return ([stmt], [])

        else:
            print 'internal error:Tiling: unknown type of statement: %s' % stmt.__class__.__name__
            sys.exit(1)
        
    

    #----------------------------------------------

    def __startTiling(self, stmt, new_integer_vars):
        '''Find loops to be tiled and apply loop-tiling transformation on each of them'''

        if isinstance(stmt, ast.ExpStmt):
            return stmt

        elif isinstance(stmt, ast.CompStmt):
            stmt.stmts = [self.__startTiling(s, new_integer_vars) for s in stmt.stmts]
            return stmt

        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__startTiling(stmt.true_stmt, new_integer_vars)
            if stmt.false_stmt:
                stmt.false_stmt = self.__startTiling(stmt.false_stmt, new_integer_vars)
            return stmt

        elif isinstance(stmt, ast.ForStmt):
            tiled_stmts, untiled_stmts = self.__tile(stmt, new_integer_vars, [], [])
            if len(untiled_stmts) > 0:
                print 'internal error:Tiling: untiled statements must be empty at this point'
                sys.exit(1)
            if len(tiled_stmts) != 1:
                print 'internal error:Tiling: only one tiled statement is expected'
                sys.exit(1)
            return tiled_stmts[0]

        else:
            print 'internal error:Tiling: unknown type of statement: %s' % stmt.__class__.__name__
            sys.exit(1)

    #----------------------------------------------

    def transform(self, code_stmts):
        '''To apply loop-tiling transformation on the given code'''

        # perform loop tiling on each statement
        new_integer_vars = []
        tiled_code_stmts = [self.__startTiling(s, new_integer_vars) for s in code_stmts]

        # return the tiled code statements
        return tiled_code_stmts

