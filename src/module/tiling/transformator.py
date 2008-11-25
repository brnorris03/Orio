#
# The implementation of the code transformator that performs loop tiling
#

import sys
import ast, ast_util

#-------------------------------------------------

class Transformator:
    '''The code transformator that performs loop tiling'''

    def __init__(self, perf_params):
        '''To instantiate a code transformator'''

        self.perf_params = perf_params
        self.ast_util = ast_util.ASTUtil()

    #----------------------------------------------

    def __getInterTileLoopIndexName(self, index_name):
        '''Generate a new index name used as the inter-tile loop index name'''
        return index_name + 't'

    #----------------------------------------------

    def __createIntraTileLoop(self, index_id, tile_size_id, stride_exp, loop_body):
        '''
        Generate an intra-tile loop statement that corresponds to the given input arguments
          for (i=it; i<=it+(Ts-St); i+=St)
            <loop-body>
        '''

        lbound_exp = tile_size_id
        e = ast.BinOpExp(tile_size_id, stride_exp, ast.BinOpExp.SUB)
        inter_tile_index_id = ast.IdentExp(self.__getInterTileLoopIndexName(index_id.name))
        ubound_exp = ast.BinOpExp(inter_tile_index_id, ast.ParenthExp(e), ast.BinOpExp.ADD)
        return self.ast_util.createForLoop(index_id, lbound_exp, ubound_exp, stride_exp, loop_body)

    #----------------------------------------------

    def __createInterTileLoop(self, index_id, tile_size_id, lbound_exp, ubound_exp, loop_body):
        '''
        Generate an inter-tile loop statement that corresponds to the given input arguments
          for (it=lb; it<=ub-(Ts-1); it+=Ts)
            <loop-body>
        '''

        inter_tile_index_id = ast.IdentExp(self.__getInterTileLoopIndexName(index_id.name))
        e = ast.BinOpExp(tile_size_id, ast.NumLitExp(1, ast.NumLitExp.INT), ast.BinOpExp.SUB)
        ubound_exp = ast.BinOpExp(ubound_exp, ast.ParenthExp(e), ast.BinOpExp.SUB)
        return self.ast_util.createForLoop(inter_tile_index_id, lbound_exp, ubound_exp,
                                           tile_size_id, loop_body)

    #----------------------------------------------

    def __tileLoopBody(self, stmt, tile_info_table, new_integer_vars, outer_tiled_loop_seq):
        '''Apply loop tiling to the given loop body statement'''

        if not isinstance(stmt, ast.CompStmt):
            print 'error:Tiling: statement is not a compound statement'
            sys.exit(1)

        #XXX
        

        if isinstance(stmt, ast.ExpStmt):
            return stmt

        elif isinstance(stmt, ast.CompStmt):
            stmt.stmts = [self.__tile(s, tile_info_table, outer_index_names, new_integer_vars)
                          for s in stmt.stmts]
            return stmt

        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__tile(stmt.true_stmt, tile_info_table,
                                         outer_index_names, new_integer_vars)
            if stmt.false_stmt:
                stmt.false_stmt = self.__tile(stmt.false_stmt, tile_info_table,
                                              outer_index_names, new_integer_vars)
            return stmt

        elif isinstance(stmt, ast.ForStmt):

            # extract information about this loop structure
            for_loop_info = self.ast_util.getForLoopInfo(stmt)
            (index_id, lbound_exp, ubound_exp, stride_exp, loop_body) = for_loop_info

            # perform tiling if this loop is to be tiled
            if index_id.name in tile_info_table:
                tiled_loop = self.__tileLoop(stmt, tile_info_table,
                                             outer_index_names, new_integer_vars)
                return tiled_loop
                
            # don't perform tiling on this loop
            else:
                stmt.stmt = self.__tile(stmt.stmt, tile_info_table,
                                        outer_index_names, new_integer_vars)
                return stmt

        else:
            print 'internal error:Tiling: unknown type of AST: %s' % stmt.__class__.__name__
            sys.exit(1)

    #----------------------------------------------

    def __tileLoop(self, stmt, tile_info_table, new_integer_vars, outer_tiled_loop_seq):
        '''Apply tiling to the given loop statement'''

        # extract information about this loop structure
        for_loop_info = self.ast_util.getForLoopInfo(stmt)
        (index_id, lbound_exp, ubound_exp, stride_exp, loop_body) = for_loop_info
        
        # get the tiling information that corresponds to this loop
        index_name = index_id.name
        tile_size_name, tile_size_value = tile_info_table[index_name]

        # recursively apply tiling to the loop body
        n_outer_tiled_loop_seq = n_outer_tiled_loop_seq + [index_name]
        tiled_loop_body = self.__tileLoopBody(stmt.stmt, tile_info_table, new_integer_vars,
                                              n_outer_tiled_loop_seq)

        # generate the tiled loop
        tile_size_id = ast.IdentExp(tile_size_name)
        tiled_loop = self.__createInterTileLoop(index_id, tile_size_id, lbound_exp,
                                                ubound_exp, tiled_loop_body)

        # return the tiled loop
        return tiled_loop

    #----------------------------------------------

    def __tile(self, stmt, tile_info_table, new_integer_vars):
        '''
        Apply tiling to the loops with matching index names as specified in the tiling information
        '''

        if isinstance(stmt, ast.ExpStmt):
            return stmt

        elif isinstance(stmt, ast.CompStmt):
            stmt.stmts = [self.__tile(s, tile_info_table, new_integer_vars) for s in stmt.stmts]
            return stmt

        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__tile(stmt.true_stmt, tile_info_table, new_integer_vars)
            if stmt.false_stmt:
                stmt.false_stmt = self.__tile(stmt.false_stmt, tile_info_table, new_integer_vars)
            return stmt

        elif isinstance(stmt, ast.ForStmt):

            # extract information about this loop structure
            for_loop_info = self.ast_util.getForLoopInfo(stmt)
            (index_id, lbound_exp, ubound_exp, stride_exp, loop_body) = for_loop_info

            # perform tiling if this loop is to be tiled
            if index_id.name in tile_info_table:
                tiled_loop = self.__tileLoop(stmt, tile_info_table, new_integer_vars, [])
                return tiled_loop
                
            # don't perform tiling on this loop
            else:
                stmt.stmt = self.__tile(stmt.stmt, tile_info_table, new_integer_vars)
                return stmt

        else:
            print 'internal error:Tiling: unknown type of AST: %s' % stmt.__class__.__name__
            sys.exit(1)

    #----------------------------------------------

    def transform(self, code_stmts, tile_info_list):
        '''To apply loop-tiling transformation on the given code'''

        # put all tiling information into a hashtable
        tile_info_table = {}
        for index_name, tile_size_name, tile_size_value in tile_info_list:
            if tile_size_value != None:
                tile_size_value = eval(tile_size_value, self.perf_params)
            tile_info_table[index_name] = (tile_size_name, tile_size_value)

        # normalize the format of all for-loops
        code_stmts = [ast_util.ASTUtil().normalizeLoopFormat(s) for s in code_stmts]

        # perform loop tiling on each statement
        new_integer_vars = []
        tiled_code_stmts = [self.__tile(s, tile_info_table, new_integer_vars) for s in code_stmts]

        # return the tiled code statements
        return tiled_code_stmts


