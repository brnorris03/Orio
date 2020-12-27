#
# Contain the transformation procedure
#

import sys
from orio.main.util.globals import *
import orio.module.loop.ast, orio.module.loop.ast_lib.common_lib, orio.module.loop.ast_lib.forloop_lib
import orio.module.loop.submodule.tile.tile
import orio.module.loop.submodule.permut.permut
import orio.module.loop.submodule.regtile.regtile
import orio.module.loop.submodule.unrolljam.unrolljam
import orio.module.loop.submodule.scalarreplace.scalarreplace
import orio.module.loop.submodule.boundreplace.boundreplace
import orio.module.loop.submodule.pragma.pragma
import orio.module.loop.submodule.arrcopy.arrcopy
import orio.module.loop.submodule.cuda.cuda

#-----------------------------------------

class Transformation:
    '''Code transformation implementation'''

    def __init__(self, tiles, permuts, regtiles, ujams, scalarrep, boundrep,
                 pragma, openmp, vector, arrcopy, cuda, stmt):
        '''Instantiate a code transformation object'''

        self.tiles = tiles
        self.permuts = permuts
        self.regtiles = regtiles
        self.ujams = ujams
        self.scalarrep = scalarrep
        self.boundrep = boundrep
        self.pragma = pragma
        self.openmp = openmp
        self.vector = vector
        self.arrcopy = arrcopy
        self.cuda = cuda
        self.stmt = stmt
        self.label = stmt.label

        self.counter = 1
        self.prefix = 'cbv_'
        
        self.flib = orio.module.loop.ast_lib.forloop_lib.ForLoopLib()
        self.clib = orio.module.loop.ast_lib.common_lib.CommonLib()

        self.tile_smod = orio.module.loop.submodule.tile.tile.Tile()
        self.perm_smod = orio.module.loop.submodule.permut.permut.Permut()
        self.regt_smod = orio.module.loop.submodule.regtile.regtile.RegTile()
        self.ujam_smod = orio.module.loop.submodule.unrolljam.unrolljam.UnrollJam()
        self.srep_smod = orio.module.loop.submodule.scalarreplace.scalarreplace.ScalarReplace()
        self.brep_smod = orio.module.loop.submodule.boundreplace.boundreplace.BoundReplace()
        self.prag_smod = orio.module.loop.submodule.pragma.pragma.Pragma()
        self.acop_smod = orio.module.loop.submodule.arrcopy.arrcopy.ArrCopy()
        self.cuda_smod = orio.module.loop.submodule.cuda.cuda.CUDA()

    #----------------------------------------------------------

    def __tile(self, stmt, tinfo):
        '''To apply loop tiling'''

        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            debug("Not tiling ExpStmt", obj=self)

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            debug("Tiling CompStmt", obj=self)
            stmt.stmts = [self.__tile(s, tinfo) for s in stmt.stmts]

        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            debug("Tiling IfStmt", obj=self)
            stmt.true_stmt = self.__tile(stmt.true_stmt, tinfo)
            
            if stmt.false_stmt:
                stmt.false_stmt = self.__tile(stmt.false_stmt, tinfo)

        elif isinstance(stmt, orio.module.loop.ast.ForStmt):

            debug("Tiling ForStmt " + str(tinfo), obj=self)
            # recursively transform the loop body
            stmt.stmt = self.__tile(stmt.stmt, tinfo)

            # apply tiling if this is the loop to be tiled
            for_loop_info = self.flib.extractForLoopInfo(stmt)
            index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info
            lid, tsize, tindex = tinfo
            if lid == index_id.name:
                stmt = self.tile_smod.tile(tsize, tindex, stmt)

            # return this loop statement

        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            debug("Tiling TransformStmt: THIS SHOULD NOT HAPPEN!", obj=self)
            err('orio.module.loop.submodule.composite.transformation internal error (__tile): unprocessed transform statement')
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            debug("Not tiling NewAST", obj=self)

        elif isinstance(stmt, orio.module.loop.ast.Comment):
            debug("Not tiling Comment", obj=self)

        else:
            err('orio.module.loop.submodule.composite.transformation internal error (__tile): unexpected AST type: "%s"' % stmt.__class__)
        debug('Returning from _tile %s' % repr(stmt),obj=self)
        return stmt
    
    #----------------------------------------------------------
    def __cudify(self, stmt, targs):
        if not stmt: return None
        
        if isinstance(stmt, orio.module.loop.ast.ForStmt):
            stmt = self.cuda_smod.cudify(stmt, targs)
        elif isinstance(stmt, orio.module.loop.ast.VarDecl):
            pass
        elif isinstance(stmt, orio.module.loop.ast.VarDeclInit):
            pass
        elif isinstance(stmt, orio.module.loop.ast.ExpStmt):
            pass
        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            tstmts = []
            for s in stmt.stmts:
                t = self.__cudify(s, targs)
                tstmts.append(t)
            stmt.stmts = tstmts
        else:
            err('orio.module.loop.submodule.composite.transformation internal error (__cudify): unexpected AST type: "%s"' % stmt.__class__)
        return stmt
    
    
        
    #----------------------------------------------------------

    def __unrollJam(self, stmt, tinfos):
        '''To apply loop unroll/jamming'''
        if not stmt: return None

        #debug('orio.module.loop.submodule.composite.transformation: entering __unrollJam, stmt %s: %s' % (stmt.__class__, stmt))
        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            return (stmt, [])

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            tstmts = []
            unrolled_loop_infos = []
            for s in stmt.stmts:
                t,l = self.__unrollJam(s, tinfos)
                tstmts.append(t)
                unrolled_loop_infos.extend(l)
            stmt.stmts = tstmts
            return (stmt, unrolled_loop_infos)
            
        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            unrolled_loop_infos = []
            t,l = self.__unrollJam(stmt.true_stmt, tinfos)
            stmt.true_stmt = t
            unrolled_loop_infos.extend(l)
            if stmt.false_stmt:
                t,l = self.__unrollJam(stmt.false_stmt, tinfos)
                stmt.false_stmt = t
                unrolled_loop_infos.extend(l)
            return (stmt, unrolled_loop_infos)

        elif isinstance(stmt, orio.module.loop.ast.ForStmt):
            t,l = self.__unrollJam(stmt.stmt, tinfos)
            stmt.stmt = t
            unrolled_loop_infos = l[:]

            for_loop_info = self.flib.extractForLoopInfo(stmt)
            index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info
            
            ufactor = -1
            for lid, uf in tinfos:
                if lid == index_id.name:
                    ufactor = uf
                    break

            if ufactor > 1:
                do_jamming = True
                for lbexp, ubexp, stexp in unrolled_loop_infos:
                    do_jamming = (do_jamming and 
                                  (not self.clib.containIdentName(lbexp, index_id.name)) and
                                  (not self.clib.containIdentName(ubexp, index_id.name)) and
                                  (not self.clib.containIdentName(stexp, index_id.name)))
                parallelize = False
                stmt = self.ujam_smod.unrollAndJam(ufactor, do_jamming, stmt, parallelize)
                unrolled_loop_infos.append((lbound_exp, ubound_exp, stride_exp))

            return (stmt, unrolled_loop_infos)

        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.composite.transformation internal error: unprocessed transform statement')            

        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return (stmt, [])
        
        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return (stmt, [])

        else:
            err('orio.module.loop.submodule.composite.transformation internal error (__unrollJam): unexpected AST type: "%s"' % stmt.__class__.__name__)            
        
    #----------------------------------------------------------
    
    def __insertPragmas(self, stmt, tinfo):
        '''To insert pragma directives'''
        
        if not stmt: return None
        
        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            return stmt

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            nstmts = []
            for s in stmt.stmts:
                is_comp_before = isinstance(s, orio.module.loop.ast.CompStmt)
                ns = self.__insertPragmas(s, tinfo)
                is_comp_after = isinstance(ns, orio.module.loop.ast.CompStmt)
                if not is_comp_before and is_comp_after:
                    nstmts.extend(ns.stmts)
                else:
                    nstmts.append(ns)
            stmt.stmts = nstmts
            return stmt
            
        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            stmt.true_stmt = self.__insertPragmas(stmt.true_stmt, tinfo)
            if stmt.false_stmt:
                stmt.false_stmt = self.__insertPragmas(stmt.false_stmt, tinfo)
            return stmt
                
        elif isinstance(stmt, orio.module.loop.ast.ForStmt):

            # recursively transform the loop body
            stmt.stmt = self.__insertPragmas(stmt.stmt, tinfo)

            # apply tiling if this is the loop to be tiled
            for_loop_info = self.flib.extractForLoopInfo(stmt)
            index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info
            lid, pragmas = tinfo
            if lid == index_id.name:
                stmt = self.prag_smod.insertPragmas(pragmas, stmt)

            # return this loop statement
            return stmt

        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.composite.transformation internal error (__insertPragmas): unprocessed transform statement')            
                                    
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return stmt
        
        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return stmt

        else:
            err('orio.module.loop.submodule.composite.transformation internal error (__insertPragmas): unexpected AST type: "%s"' % stmt.__class__)
                   
    #----------------------------------------------------------

    def __replaceBoundsInsertPrags(self, stmt, pragmas):
        '''Replace loop bounds with scalars, then insert pragmas right before the loop'''

        if not stmt: return None
        # get the loop structure
        for_loop_info = self.flib.extractForLoopInfo(stmt)
        index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info

        # replace complex loop bounds with scalars
        decls = []
        asgns = []
        if lbound_exp and self.clib.isComplexExp(lbound_exp):
            intmd = orio.module.loop.ast.IdentExp(self.prefix + str(self.counter))
            self.counter += 1
            stmt.init.rhs = intmd.replicate()
            decls.append(orio.module.loop.ast.VarDecl('register int', [intmd.name]))
            asgn = orio.module.loop.ast.BinOpExp(intmd.replicate(),
                                            lbound_exp.replicate(),
                                            orio.module.loop.ast.BinOpExp.EQ_ASGN)
            asgn = orio.module.loop.ast.ExpStmt(asgn)
            asgns.append(asgn)
        if ubound_exp and self.clib.isComplexExp(ubound_exp):
            intmd = orio.module.loop.ast.IdentExp(self.prefix + str(self.counter))
            self.counter += 1
            stmt.test.rhs = intmd.replicate()
            if len(decls) > 0:
                decls[0].var_names.append(intmd.name)
            else:
                decls.append(orio.module.loop.ast.VarDecl('register int', [intmd.name]))
            asgn = orio.module.loop.ast.BinOpExp(intmd.replicate(),
                                            ubound_exp.replicate(),
                                            orio.module.loop.ast.BinOpExp.EQ_ASGN)
            asgn = orio.module.loop.ast.ExpStmt(asgn)
            asgns.append(asgn)      
        
        # generate the transformed loop
        stmt = self.prag_smod.insertPragmas(pragmas, stmt)
        stmt.stmts = decls + asgns + stmt.stmts
        
        # return this loop statement
        return stmt

    #----------------------------------------------------------
    
    def __insertOpenMPPragmas(self, stmt, tinfo):
        '''To insert OpenMP pragma directives (on outermost loops only)'''
        
        if not stmt: return None
        
        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            return stmt

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            nstmts = []
            for s in stmt.stmts:
                is_comp_before = isinstance(s, orio.module.loop.ast.CompStmt)
                ns = self.__insertOpenMPPragmas(s, tinfo)
                is_comp_after = isinstance(ns, orio.module.loop.ast.CompStmt)
                if not is_comp_before and is_comp_after:
                    nstmts.extend(ns.stmts)
                else:
                    nstmts.append(ns)
            stmt.stmts = nstmts
            return stmt
            
        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            stmt.true_stmt = self.__insertOpenMPPragmas(stmt.true_stmt, tinfo)
            if stmt.false_stmt:
                stmt.false_stmt = self.__insertOpenMPPragmas(stmt.false_stmt, tinfo)
            return stmt
                
        elif isinstance(stmt, orio.module.loop.ast.ForStmt):

            # get the loop structure
            for_loop_info = self.flib.extractForLoopInfo(stmt)
            index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info
            
            # check if the initialization, test, and iteration variables exist
            if lbound_exp == None or ubound_exp == None or stride_exp == None:
                return stmt

            # replace loop bounds with scalars, and then insert openMP pragmas before the loop
            pragmas, = tinfo
            stmt = self.__replaceBoundsInsertPrags(stmt, pragmas)

            # return the transformed loop
            return stmt

        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.composite.transformation internal error (__insertOpenMPPragmas): unprocessed transform statement')
                                                
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return stmt
        
        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return stmt          

        else:
            err('orio.module.loop.submodule.composite.transformation internal error (__insertOpenMPPragmas): unexpected AST type: "%s"' % stmt.__class__)            
        
    #----------------------------------------------------------
        
    def __insertVectorPragmas(self, stmt, tinfo):
        '''To insert vectorization pragma directives (on innermost loops only)'''

        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            return stmt

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            nstmts = []
            for s in stmt.stmts:
                is_comp_before = isinstance(s, orio.module.loop.ast.CompStmt)
                ns = self.__insertVectorPragmas(s, tinfo)
                is_comp_after = isinstance(ns, orio.module.loop.ast.CompStmt)
                if not is_comp_before and is_comp_after:
                    nstmts.extend(ns.stmts)
                else:
                    nstmts.append(ns)
            stmt.stmts = nstmts
            return stmt
        
        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            stmt.true_stmt = self.__insertVectorPragmas(stmt.true_stmt, tinfo)
            if stmt.false_stmt:
                stmt.false_stmt = self.__insertVectorPragmas(stmt.false_stmt, tinfo)
            return stmt
    
        elif isinstance(stmt, orio.module.loop.ast.ForStmt):

            # apply recursion on the loop body
            stmt.stmt = self.__insertVectorPragmas(stmt.stmt, tinfo)

            # no transformation if it is not the inner most loop
            if self.flib.hasInnerLoop(stmt.stmt):
                return stmt

            # replace loop bounds with scalars, and then insert openMP pragmas before the loop
            pragmas, = tinfo
            stmt = self.__replaceBoundsInsertPrags(stmt, pragmas)

            # return the transformed loop
            return stmt

        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.composite.transformation internal error (__insertVectorPragmas): unprocessed transform statement')
                                                
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return stmt
        
        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return stmt
            
        else:
            err('orio.module.loop.submodule.composite.transformation internal error (__insertVectorPragmas): unexpected AST type: "%s"' % stmt.__class__.__name__)
            
        
    #----------------------------------------------------------
    
    def __searchLoopId(self, all_lids, input_lid):
        '''
        Given all the existing loop index names, search if there is a match with the given loop ID.
        Complain necessarily if there is no match found.
        '''

        for i in input_lid[1:]:
            if i in all_lids:
                return i
        must_exist = input_lid[0]
        if must_exist:
            err('orio.module.loop.submodule.composite.transformation (__searcLoopId): no matching loop index name in input argument: %s' %
                   (tuple(input_lid[1:]), ))            
        else:
            return None
            
    #----------------------------------------------------------

    def transform(self):
        '''To apply the composite transformations'''

        # copy the statement
        tstmt = self.stmt.replicate()

        # Use a label with the original annotation line number to identify the loop
        if not tstmt.meta.get('id') and tstmt.line_no:
            tstmt.meta['id'] = 'loop_' + str(tstmt.line_no)
        # reset counter (for variable name generation)
        self.counter = 1

        debug("Before applying tiling", obj=self)

        # apply loop tiling
        for loop_id, tsize, tindex in self.tiles:
            all_lids = self.flib.getLoopIndexNames(tstmt)
            lid = self.__searchLoopId(all_lids, loop_id)
            if lid != None:
                tinfo = (lid, tsize, tindex)
                debug('applying tiling to loop_id=%s' % str(loop_id),obj=self)
                try:
                    tstmt = self.__tile(tstmt, tinfo)
                except Exception as e:
                    err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying tiling (tsize=%s)\n' + \
                         '--> %s: %s'\
                         % (str(self.stmt.line_no), str(tsize), e.__class__.__name__, e.message))
                debug('SUCCESS: applying tiling to loop_id=%s' % str(loop_id), obj=self)

        debug("After applying tiling", obj=self)
        # apply loop permutation/interchange
        try: 
            for seq in self.permuts:
                debug('applying loop permutation/interchange',obj=self)
                tstmt = self.perm_smod.permute(seq, tstmt)
                debug('SUCCESS: applying loop permutation/interchange', obj=self)
        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'loop permutations: "%s"\npermutation annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.perm_smod.__class__, str(self.permuts), e.__class__.__name__, e.message))

        # apply array-copy optimization
        debug('applying array copy')
        try: 
            for do_acopy, aref, suffix, dtype, dimsizes in self.arrcopy:
                if not do_acopy:
                    dimsizes = [1] * len(dimsizes)
                debug('applying %s' % self.acop_smod.__class__.__name__, obj=self)
                tstmt = self.acop_smod.optimizeArrayCopy(aref, suffix, dtype, dimsizes, tstmt)
                debug('SUCCESS: applying %s' % self.acop_smod.__class__.__name__, obj=self)
        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'array copy: "%s"\narray copy annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.acop_smod.__class__, str(self.arrcopy), e.__class__.__name__, e.message))

        # apply register tiling
        loops, ufactors = self.regtiles
        try:
            if len(loops) > 0:
                debug('applying register tiling', obj=self)
                tstmt = self.regt_smod.tileForRegs(loops, ufactors, tstmt)
        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'register tiling: "%s"\nregtile annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.regt_smod.__class__, str(self.regtiles), e.__class__.__name__, e.message))
            if len(loops) > 0: debug('SUCCESS: applying register tiling', obj=self)


        # apply unroll/jamming
        try:
            loops, ufactors = self.ujams
            tinfos = []
            for loop_id, ufactor in zip(loops, ufactors):
                all_lids = self.flib.getLoopIndexNames(tstmt)
                lid = self.__searchLoopId(all_lids, (False, loop_id))
                if lid != None and ufactor > 1:
                    tinfos.append((lid, ufactor))
            if len(tinfos) > 0:
                debug('applying unroll/jam')
                tstmt,_ = self.__unrollJam(tstmt, tinfos)
        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'loop unrolling/jamming: "%s"\nunroll/jam annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.ujam_smod.__class__, str(self.ujams), e.__class__, e))

        # apply scalar replacement
        do_scalarrep, dtype, prefix = self.scalarrep
        try:
            if do_scalarrep:
                debug('applying scalar replacement', obj=self)
                tstmt = self.srep_smod.replaceScalars(dtype, prefix, tstmt)
        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'scalar replacement: "%s"\nscalar replacement annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.srep_smod.__class__, str(self.scalarrep), e.__class__, e))
        if do_scalarrep: debug('SUCCESS: applying scalar replacement', obj=self)

        # apply bound replacement
        do_boundrep, lprefix, uprefix = self.boundrep
        try:
            if do_boundrep:
                debug('applying bounds replacement', obj=self)
                tstmt = self.brep_smod.replaceBounds(lprefix, uprefix, tstmt)
        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'bound replacement: "%s"\nbounds annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.srep_smod.__class__, str(self.boundrep), e.__class__, e))
        if do_boundrep: debug('SUCCESS: applying bounds replacement', obj=self)

        # insert pragma directives
        try:
            debug('applying pragmas', obj=self)
            for loop_id, pragmas in self.pragma:
                all_lids = self.flib.getLoopIndexNames(tstmt)
                lid = self.__searchLoopId(all_lids, loop_id)
                if lid != None:
                    tinfo = (lid, pragmas)
                    tstmt = self.__insertPragmas(tstmt, tinfo)
        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'pragma directives: "%s"\npragma annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.srep_smod.__class__, str(self.pragma), e.__class__, e))
        debug('SUCCESS: applying pragmas', obj=self)

        # insert openmp directives (apply only on outermost loops)
        try:
            do_openmp, pragmas = self.openmp
            if do_openmp:
                debug('applying openmp',obj=self)
                tinfo = (pragmas, )
                tstmt = self.__insertOpenMPPragmas(tstmt, tinfo)
        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'openmp directives: "%s"\nopenmp annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.srep_smod.__class__, str(self.openmp), e.__class__, e))
        if do_openmp: debug('SUCCESS: applying openmp', obj=self)

        # insert vectorization directives (apply only on innermost loops)
        try:
            do_vector, pragmas = self.vector
            if do_vector:
                debug('applying vectorization (inserting directives)',obj=self)
                tinfo = (pragmas, )
                tstmt = self.__insertVectorPragmas(tstmt, tinfo)

        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'vectorization: "%s"\nvector annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.srep_smod.__class__, str(self.vector), e.__class__, e))
        if do_vector: debug('SUCCESS: applying vectorization (inserting directives)', obj=self)

        # apply cuda transformation
        try:
            threadCount, cacheBlocks, pinHost, streamCount = self.cuda
            if threadCount:
                debug('applying cuda', obj=self)
                targs = {'threadCount':threadCount, 'cacheBlocks':cacheBlocks, 'pinHostMem':pinHost, 'streamCount':streamCount, 'domain':None, 'dataOnDevice':False, 'blockCount':5, 'unrollInner':0, 'preferL1Size':16}
                tstmt = self.__cudify(tstmt, targs)

        except Exception as e:
            err('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying ' +
                 'cuda: "%s"\ncuda annotation: %s\n --> %s: %s' \
                 % (self.stmt.line_no, self.cuda_smod.__class__, str(self.cuda), e.__class__, e),doexit=True)
            import traceback
            import sys
            raise TransformationException('orio.module.loop.submodule.composite.transformation:%s: encountered an error in applying cuda: "%s"\ncuda annotation: %s\n --> %s: %s\n %s\n' % (self.stmt.line_no, self.cuda_smod.__class__.__name__, self.cuda, e.__class__.__name__, e, traceback.format_exc()))
        if threadCount: debug('SUCCESS: applying cuda', obj=self)
        # return the transformed statement

        debug('orio.module.loop.submodule.composite.transformation: End of transform() method',obj=self)
        return tstmt


