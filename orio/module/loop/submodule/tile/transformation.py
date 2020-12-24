#
# Contain the transformation procedure
#

import sys
from orio.main.util.globals import *
import orio.module.loop.ast, orio.module.loop.ast_lib.constant_folder, orio.module.loop.ast_lib.forloop_lib

#-----------------------------------------

class Transformation:
    '''Code transformation'''

    def __init__(self, tsize, tindex, stmt, language='C'):
        '''To instantiate a code transformation object'''

        self.tsize = tsize
        self.tindex = tindex
        self.language = language
        self.stmt = stmt
        self.localvars = set([])
        
        self.flib = orio.module.loop.ast_lib.forloop_lib.ForLoopLib()
        self.cfolder = orio.module.loop.ast_lib.constant_folder.ConstFolder()
        
    #----------------------------------------------------------

    def transform(self):
        '''
        To tile the given loop structure. The resulting tiled loop will look like this.
        
        for (ii=LB; ii<=UB; ii+=Ti)
          for (i=ii; i<=min(UB,ii+Ti-ST); i+=ST)
            <loop-body>

        Such tiled code avoids the uses of division/multiplication operations. However, some
        compilers (e.g. ICC -fast) cannot finish its compilation when multiple levels of tiling
        are applied.
        '''

        debug("tile.Transformation starting",obj=self)

        # get rid of compound statement that contains only a single statement
        while isinstance(self.stmt, orio.module.loop.ast.CompStmt) and len(self.stmt.stmts) == 1:
            self.stmt = self.stmt.stmts[0]
                                
        # extract for-loop structure
        for_loop_info = self.flib.extractForLoopInfo(self.stmt)
        index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info

        # check the tile index name
        if self.tindex == index_id.name:
            err(('orio.module.loop.submodule.tile.transformation:%s: the tile index name must be different from the new tiled ' + \
                    'loop index name: "%s"') % (index_id.line_no, self.tindex))
        
        # when tile size = 1, no transformation will be applied
        if self.tsize == 1:
            debug("tile.Transformation returning tsize=1", obj=self)
            try:
                st = self.flib.createForLoop(index_id, lbound_exp, ubound_exp,
                                            stride_exp, loop_body)
            except Exception, e:
                err("orio.module.loop.submodule.tile.transformation:%s: error creating for loop for tile size 1\n --> %s: %s" %
                    (stride_exp.line_no, e.__class__.__name__, e.message))
            return st

        # evaluate the stride expression
        try:
            stride_val = eval(str(stride_exp))
        except Exception, e:
            err('orio.module.loop.submodule.tile.transformation:%s: failed to evaluate expression: "%s"\n --> %s: %s' %
                   (stride_exp.line_no, stride_exp,e.__class__.__name__, e.message))
        if not isinstance(stride_val, int) or stride_val <= 0:
            err('orio.module.loop.submodule.tile.transformation:%s: loop stride size must be a positive integer: %s' %
                   (stride_exp.line_no, stride_exp))

        # check whether tile_size % stride == 0
        if self.tsize % stride_val != 0:
            err('orio.module.loop.submodule.tile.transformation:%s: tile size (%s) must be divisible by the stride value (%s)'
                   % (stride_exp.line_no, str(self.tsize), str(stride_val)))

        # sanity check whether tile_size > stride
        if self.tsize <= stride_val:
            # Issue a warning, then return the untransformed statement without performing tiling
            msg = 'tile size ' + str(self.tsize) + ' must be greater than the stride value ' + str(stride_val) + \
                '; tile index = ' + str(self.tindex)
            warn('orio.module.loop.submodule.tile.transformation: ' + msg)
            return self.stmt.replicate()

        # create the tile index name
        tindex_id = orio.module.loop.ast.IdentExp(self.tindex)

        # for the inter-tiling loop (i.e. outer loop)
        # compute lower bound --> LB' = LB
        tile_lbound_exp = lbound_exp.replicate()
        
        # compute upper bound --> UB' = UB
        tile_ubound_exp = ubound_exp.replicate()
        
        # compute stride --> ST' = tsize
        tile_stride_exp = orio.module.loop.ast.NumLitExp(self.tsize, orio.module.loop.ast.NumLitExp.INT) 

        # for the intra-tile loop (i.e. inner loop)
        # compute lower bound --> LB' = tindex
        itile_lbound_exp = tindex_id.replicate()

        # compute upper bound --> UB' = min(UB, tindex+tsize-ST)
        it1 = orio.module.loop.ast.BinOpExp(orio.module.loop.ast.NumLitExp(self.tsize,
                                                                 orio.module.loop.ast.NumLitExp.INT),
                                       stride_exp.replicate(),
                                       orio.module.loop.ast.BinOpExp.SUB)
        it2 = orio.module.loop.ast.BinOpExp(tindex_id.replicate(), it1, orio.module.loop.ast.BinOpExp.ADD)
        it2 = self.cfolder.fold(it2)
        itile_ubound_exp = orio.module.loop.ast.FunCallExp(orio.module.loop.ast.IdentExp('min'),
                                                      [ubound_exp.replicate(), it2])

        # compute stride --> ST' = ST
        itile_stride_exp = stride_exp.replicate()

        # generate the transformed statement
        transformed_stmt = self.flib.createForLoop(tindex_id, tile_lbound_exp,
                                                   tile_ubound_exp, tile_stride_exp,
                                                   self.flib.createForLoop(index_id,
                                                                           itile_lbound_exp,
                                                                           itile_ubound_exp,
                                                                           itile_stride_exp,
                                                                           loop_body))

        # return the transformed statement
        debug("tile.Transformation returning (final)",obj=self)
        return transformed_stmt
             
