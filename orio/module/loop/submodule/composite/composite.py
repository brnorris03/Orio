#
# Loop transformation submodule.that implements a combination of various loop transformations.
#

import sys
import orio.module.loop.submodule.submodule, transformation
import orio.module.loop.submodule.tile.tile
import orio.module.loop.submodule.permut.permut
import orio.module.loop.submodule.regtile.regtile
import orio.module.loop.submodule.unrolljam.unrolljam
import orio.module.loop.submodule.scalarreplace.scalarreplace
import orio.module.loop.submodule.boundreplace.boundreplace
import orio.module.loop.submodule.pragma.pragma
import orio.module.loop.submodule.arrcopy.arrcopy
import orio.module.loop.submodule.cuda.cuda
from orio.main.util.globals import *

#---------------------------------------------------------------------

class Composite(orio.module.loop.submodule.submodule.SubModule):
    '''The composite loop transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='C', tinfo = None):
        '''To instantiate a composite loop transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)

        # transformation submodule.
        self.tile_smod = orio.module.loop.submodule.tile.tile.Tile()
        self.perm_smod = orio.module.loop.submodule.permut.permut.Permut()
        self.regt_smod = orio.module.loop.submodule.regtile.regtile.RegTile()
        self.ujam_smod = orio.module.loop.submodule.unrolljam.unrolljam.UnrollJam()
        self.srep_smod = orio.module.loop.submodule.scalarreplace.scalarreplace.ScalarReplace()
        self.brep_smod = orio.module.loop.submodule.boundreplace.boundreplace.BoundReplace()
        self.prag_smod = orio.module.loop.submodule.pragma.pragma.Pragma()
        self.acop_smod = orio.module.loop.submodule.arrcopy.arrcopy.ArrCopy()
        self.cuda_smod = orio.module.loop.submodule.cuda.cuda.CUDA()

    #-----------------------------------------------------------------

    def __readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        TILE = 'tile'
        PERMUT = 'permut'
        REGTILE = 'regtile'
        UJAM = 'unrolljam'
        SCALARREP = 'scalarreplace'
        BOUNDREP = 'boundreplace'
        PRAGMA = 'pragma'
        OPENMP = 'openmp'
        VECTOR = 'vector'
        ARRCOPY = 'arrcopy'
        CUDA = 'cuda'

        # all expected transformation arguments
        tiles = ([], None)
        permuts = ([], None)
        regtiles = (([],[]), None)
        ujams = (([],[]), None)
        scalarrep = (False, None)
        boundrep = (False, None)
        pragma = ([], None)
        openmp = ((False, ''), None)
        vector = ((False, ''), None)
        arrcopy = ([], None)
        cuda = ((None, False, False, None), None)

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:
            
            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception, e:
                err('orio.module.loop.submodule.composite.composite: %s: failed to evaluate the argument expression: %s\n --> %s: %s' %
                     (line_no, rhs,e.__class__.__name__, e))

            # update transformation arguments
            if aname == TILE:
                tiles = (rhs, line_no)
            elif aname == PERMUT:
                permuts = (rhs, line_no)
            elif aname == REGTILE:
                regtiles = (rhs, line_no)
            elif aname == UJAM:
                ujams = (rhs, line_no)
            elif aname == SCALARREP:
                scalarrep = (rhs, line_no)
            elif aname == BOUNDREP:
                boundrep = (rhs, line_no)
            elif aname == PRAGMA:
                pragma = (rhs, line_no)
            elif aname == OPENMP:
                openmp = (rhs, line_no)
            elif aname == VECTOR:
                vector = (rhs, line_no)
            elif aname == ARRCOPY:
                arrcopy = (rhs, line_no)
            elif aname == CUDA:
                cuda = (rhs, line_no)

            # unknown argument name
            else:
                err('orio.module.loop.submodule.composite.composite: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check semantics of the transformation arguments
        (tiles, permuts, regtiles, ujams, scalarrep, boundrep,
         pragma, openmp, vector, arrcopy, cuda) = self.checkTransfArgs(tiles, permuts, regtiles, ujams,
                                                                 scalarrep, boundrep, pragma,
                                                                 openmp, vector, arrcopy, cuda)

        # return information about the transformation arguments
        return (tiles, permuts, regtiles, ujams, scalarrep, boundrep, pragma, openmp, vector, arrcopy, cuda)

    #-----------------------------------------------------------------

    def checkTransfArgs(self, tiles, permuts, regtiles, ujams, scalarrep, boundrep, pragma,
                        openmp, vector, arrcopy, cuda):
        '''Check the semantics of the given transformation arguments'''
        
        # evaluate arguments for loop tiling
        rhs, line_no = tiles
        if not isinstance(rhs, list) and not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.composite.composite: %s: tile argument must be a list/tuple: %s' % (line_no, rhs))
        targs = []
        for e in rhs:
            if (not isinstance(e, list) and not isinstance(e, tuple)) or len(e) != 3:
                err(('orio.module.loop.submodule.composite.composite:%s: element of tile argument must be in the form of ' +
                        '(<loop-id>,<tsize>,<tindex>): %s') % (line_no, e))
            loop_id, tsize, tindex = e
            loop_id = self.__convertLoopId(loop_id, line_no)
            tsize, tindex = self.tile_smod.checkTransfArgs((tsize, line_no), (tindex, line_no))
            targs.append((loop_id, tsize, tindex))
        tiles = targs

        # evaluate arguments for loop permutation/interchange
        rhs, line_no = permuts
        if not isinstance(rhs, list) and not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.composite.composite: %s: permutation argument must be a list/tuple: %s' % (line_no, rhs))
        for e in rhs:
            seq, = self.perm_smod.checkTransfArgs((e, line_no))
        permuts = rhs

        # evaluate arguments for register tiling
        rhs, line_no = regtiles
        if not isinstance(rhs, list) and not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.composite.composite: %s: register-tiling argument must be a list/tuple: %s' % (line_no, rhs))
        if len(rhs) != 2:
            err(('orio.module.loop.submodule.composite.composite:%s: register-tiling argument must be in the form of ' +
                    '(<loop-ids>,<ufactors>): %s') % (line_no, rhs))
        loops, ufactors = rhs
        loops, ufactors = self.regt_smod.checkTransfArgs((loops, line_no), (ufactors, line_no))
        regtiles = (loops, ufactors)


        # evaluate arguments for unroll/jamming
        rhs, line_no = ujams
        if not isinstance(rhs, list) and not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.composite.composite: %s: unroll/jam argument must be a list/tuple: %s' % (line_no, rhs))
        if len(rhs) != 2:
            err(('orio.module.loop.submodule.composite.composite:%s: unroll/jam argument must be in the form of ' +
                    '(<loop-ids>,<ufactors>): %s') % (line_no, rhs))
        loops, ufactors = rhs
        for lp,uf in zip(loops, ufactors):
            self.ujam_smod.checkTransfArgs((uf, line_no), (False, line_no))
        ujams = (loops, ufactors)

        # evaluate arguments for scalar replacement
        rhs, line_no = scalarrep
        if isinstance(rhs, bool) or rhs == 0 or rhs == 1:
            scalarrep = (rhs, None, None)
        else:
            if ((not isinstance(rhs, list) and not isinstance(rhs, tuple)) or len(rhs) < 1 or
                len(rhs) > 3 or (not isinstance(rhs[0], bool) and rhs[0] != 0 and rhs[0] != 1)):
                err(('orio.module.loop.submodule.composite.composite:%s: scalar replacement argument must be in the form of ' +
                        '((True|False),<dtype>,<prefix>): %s') % (line_no, rhs))
            do_scalarrep = rhs[0]
            dtype = None
            prefix = None
            if len(rhs) >= 2:
                dtype = rhs[1]
            if len(rhs) >= 3:
                prefix = rhs[2]
            dtype, prefix = self.srep_smod.checkTransfArgs((dtype, line_no), (prefix, line_no))
            scalarrep = (do_scalarrep, dtype, prefix)

        # evaluate arguments for bound replacement
        rhs, line_no = boundrep
        if isinstance(rhs, bool) or rhs == 0 or rhs == 1:
            boundrep = (rhs, None, None)
        else:
            if ((not isinstance(rhs, list) and not isinstance(rhs, tuple)) or len(rhs) < 1 or
                len(rhs) > 3 or (not isinstance(rhs[0], bool) and rhs[0] != 0 and rhs[0] != 1)):
                err(('orio.module.loop.submodule.composite.composite:%s: bound replacement argument must be in the form of ' +
                        '((True|False),<lprefix>,<uprefix>): %s') % (line_no, rhs))
            do_boundrep = rhs[0]
            lprefix = None
            uprefix = None
            if len(rhs) >= 2:
                lprefix = rhs[1]
            if len(rhs) >= 3:
                uprefix = rhs[2]
            lprefix, uprefix = self.brep_smod.checkTransfArgs((lprefix, line_no), (uprefix, line_no))
            boundrep = (do_boundrep, lprefix, uprefix)

        # evaluate arguments for pragma directives
        rhs, line_no = pragma
        if not isinstance(rhs, list) and not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.composite.composite: %s: pragma argument must be a list/tuple: %s' % (line_no, rhs))
        targs = []
        for e in rhs:
            if (not isinstance(e, list) and not isinstance(e, tuple)) or len(e) != 2:
                err(('orio.module.loop.submodule.composite.composite:%s: element of pragma directive argument must be in the form of ' +
                        '(<loop-id>,<pragma-strings>): %s') % (line_no, e))
            loop_id, pragmas = e
            loop_id = self.__convertLoopId(loop_id, line_no)
            pragmas, = self.prag_smod.checkTransfArgs((pragmas, line_no))
            targs.append((loop_id, pragmas))
        pragma = targs

        # evaluate arguments for openmp pragma directive
        rhs, line_no = openmp
        if ((not isinstance(rhs, list) and not isinstance(rhs, tuple)) or len(rhs) != 2 or
            not isinstance(rhs[0], bool)):
            err(('orio.module.loop.submodule.composite.composite:%s: element of openmp pragma directive argument must be in the form of ' +
                    '((True|False),<pragma-strings>): %s') % (line_no, rhs))
        do_openmp, pragmas = rhs
        pragmas, = self.prag_smod.checkTransfArgs((pragmas, line_no))
        openmp = do_openmp, pragmas
        
        # evaluate arguments for vectorization pragma directive
        rhs, line_no = vector
        if ((not isinstance(rhs, list) and not isinstance(rhs, tuple)) or len(rhs) != 2 or
            not isinstance(rhs[0], bool)):
            err(('orio.module.loop.submodule.composite.composite:%s: element of vectorization pragma directive argument must be in ' +
                    'the form of ((True|False),<pragma-strings>): %s') % (line_no, rhs))
        do_vector, pragmas = rhs
        pragmas, = self.prag_smod.checkTransfArgs((pragmas, line_no))
        vector = do_vector, pragmas

        # evaluate arguments for array-copy optimization
        rhs, line_no = arrcopy
        if not isinstance(rhs, list) and not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.composite.composite: %s: array-copy argument must be a list/tuple: %s' % (line_no, rhs))
        targs = []
        for e in rhs:
            if ((not isinstance(e, list) and not isinstance(e, tuple)) or len(e) > 5 or
                len(e) < 3 or not isinstance(e[0], bool)):
                err(('orio.module.loop.submodule.composite.composite:%s: element of tile argument must be in the form of ' +
                        '((True|False),<array-ref-str>,<dim-sizes>,<suffix>,<dtype>): %s') %
                       (line_no, e))
            dtype = None
            suffix = None
            if len(e) == 3:
                do_acopy, aref, dimsizes = e
            elif len(e) == 4:
                do_acopy, aref, dimsizes, suffix = e
            else:
                do_acopy, aref, dimsizes, suffix, dtype = e
            (aref, suffix,
             dtype, dimsizes)= self.acop_smod.checkTransfArgs((aref, line_no), (suffix, line_no),
                                                              (dtype, line_no), (dimsizes, line_no))
            targs.append((do_acopy, aref, suffix, dtype, dimsizes))
        arrcopy = targs

        # evaluate arguments for cuda
        rhs, line_no = cuda
        if not isinstance(rhs, tuple):
            err('orio.module.loop.submodule.cuda.cuda: %s: cuda argument must be a tuple: %s' % (line_no, rhs))
        if len(rhs) != 4:
            err(('orio.module.loop.submodule.cuda.cuda:%s: cuda argument must be in the form of ' +
                    '(<threadCount>,<cacheBlocks>,<pinHostMem>,<streamCount>): %s') % (line_no, rhs))
        cuda = rhs
        
        # return information about the transformation arguments
        return (tiles, permuts, regtiles, ujams, scalarrep, boundrep, pragma, openmp, vector, arrcopy, cuda)

    #-----------------------------------------------------------------

    def applyTransf(self, tiles, permuts, regtiles, ujams, scalarrep, boundrep,
                    pragma, openmp, vector, arrcopy, cuda, stmt):
        '''To apply a sequence of transformations'''

        # perform the composite transformations
        t = transformation.Transformation(tiles, permuts, regtiles, ujams, scalarrep,
                                        boundrep, pragma, openmp, vector, arrcopy, cuda, self.stmt)

        try:
            transformed_stmt = t.transform()
        except Exception,e:
            err('orio.module.loop.submodule.composite.composite.applyTransf : %s:%s' % (e.__class__.__name__, e.message))

        debug('SUCCESS: applyTransf on ' + self.stmt.__class__.__name__, obj=self)
        if not transformed_stmt.meta.get('id'):
            transformed_stmt.meta['id'] = 'loop_' + self.stmt.line_no

        # return the transformed statement
        return transformed_stmt
        
    #-----------------------------------------------------------------

    def __convertLoopId(self, lid, line_no):
        '''
        Convert the loop ID to a list: [True/False, id1, id2, id3, ...].
        The 'True' boolean value indicates that at least one of the loop ID must exist in the
        statement body. A 'False' value means that it is OK if no loop IDs exist in the statement
        body.
        The sequence of IDs imply that "apply optimizations on id1 (if exist), if not, apply
        optimizations on id2 (if exist), and so on and so forth".
        '''

        # check if the loop ID is well-formed
        if isinstance(lid, str):
            pass
        elif (isinstance(lid, tuple) or isinstance(lid, list)) and len(lid) > 0:
            for i in lid:
                if not isinstance(i, str):
                    err('orio.module.loop.submodule.composite.composite: %s: loop ID must be a string: %s' % (line_no, i))
        else:
            err('orio.module.loop.submodule.composite.composite: %s: invalid loop ID representation: %s' % (line_no, lid))            

        # create the loop ID abstraction
        lids = []
        if isinstance(lid, str):
            lids.append(True)
            lids.append(lid)
        elif (isinstance(lid, tuple) or isinstance(lid, list)) and len(lid) > 0:
            lids.append(isinstance(lid, tuple))
            lids.extend(lid)
        else:
            err('orio.module.loop.submodule.composite.composite internal error: '+
                'incorrect representation of the loop IDs')
        
        return lids

    #-----------------------------------------------------------------

    def transform(self):
        '''To apply various loop transformations'''
        # debugging info
        #debug("perf_params=" + str(self.perf_params), self,level=6)
        
        # read all transformation arguments
        args_info = self.__readTransfArgs(self.perf_params, self.transf_args)
        (tiles, permuts, regtiles, ujams, scalarrep,
         boundrep, pragma, openmp, vector, arrcopy, cuda) = args_info
        
        # perform all transformations
        try:
            transformed_stmt = self.applyTransf(tiles, permuts, regtiles, ujams, scalarrep, boundrep,
                                                pragma, openmp, vector, arrcopy, cuda, self.stmt)
        except Exception, e:
            err('orio.module.loop.submodule.composite.composite : error transforming "%s"\n --> %s:%s' % \
                    (self.stmt, e.__class__.__name__, e.message))

        if not transformed_stmt.meta.get('id') and self.stmt.meta.get('id'):
            transformed_stmt.meta['id'] = 'loop_' + self.stmt.meta['id']

        # return the transformed statement
        return transformed_stmt



