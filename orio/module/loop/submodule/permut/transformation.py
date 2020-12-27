#
# Contain the transformation procedure
#

import sys
import orio.module.loop.ast, orio.module.loop.ast_lib.forloop_lib
from orio.main.util.globals import *

#-----------------------------------------

class Transformation:
    '''Code transformation implementation'''

    def __init__(self, seq, stmt):
        '''To instantiate a code transformation object'''

        self.seq = seq
        self.stmt = stmt

        self.flib = orio.module.loop.ast_lib.forloop_lib.ForLoopLib()
        
    #----------------------------------------------------------

    def checkPermutSeq(self, stmt):
        '''Check if the loop permutation indexes exist'''

        inames = self.flib.getLoopIndexNames(stmt)
        for i in self.seq:
            if isinstance(i, str) and i not in inames:
                err('orio.module.loop.submodule.permut.transformation:  loop permutation with index name "%s" does not exist' % i)

    #----------------------------------------------------------

    def __getPermutNests(self, stmt):
        '''Return a list of all permutable loop nests along with the loop index names sequence'''

        # collect all permutable loop nests
        permut_nests = []
        nests = self.__collectPermutNests(stmt, permut_nests)
        for n in nests:
            if len(n) > 1:
                permut_nests.append(n)

        # take the loop index names sequence
        inames_seq = []
        for n in permut_nests:
            inames = []
            for l in n:
                index_id, lbound, ubound, stride, lbody = self.flib.extractForLoopInfo(l)
                inames.append(index_id.name)
            inames_seq.append(inames)

        # return all result information
        return (permut_nests, inames_seq)

    #----------------------------------------------------------

    def __collectPermutNests(self, stmt, pnests):
        '''To collect all permutable loop nests'''

        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            return []

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            nests = []
            for s in stmt.stmts:
                nests.extend(self.__collectPermutNests(s, pnests))
            return nests

        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            nests = []
            nests.extend(self.__collectPermutNests(stmt.true_stmt, pnests))
            if stmt.false_stmt:
                nests.extend(self.__collectPermutNests(stmt.false_stmt, pnests))
            return nests

        elif isinstance(stmt, orio.module.loop.ast.ForStmt):
            nests = []
            nests.extend(self.__collectPermutNests(stmt.stmt, pnests))
            if len(nests) == 0:
                return [[stmt]]
            elif len(nests) == 1:
                nests[0].insert(0, stmt)
                return nests
            else:
                for n in nests:
                    if len(n) > 1:
                        pnests.append(n)
                return [[stmt]]

        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.permut.transformation internal error: unprocessed transform statement')
                        
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return []
        
        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return []
            

        else:
            err('orio.module.loop.submodule.permut.transformation internal error: unexpected AST type: "%s"' % stmt.__class__.__name__)
                        
    #-------------------------------------------------

    def __interchangeTwoLoops(self, loop1, loop2):
        '''Swap the control expressions between the given two for-loops'''

        init1 = loop1.init
        test1 = loop1.test
        iter1 = loop1.iter
        loop1.init = loop2.init
        loop1.test = loop2.test
        loop1.iter = loop2.iter
        loop2.init = init1
        loop2.test = test1
        loop2.iter = iter1

    #-------------------------------------------------

    def __permuteLoops(self, lnest, from_seq, to_seq):
        '''Permute the loop nests based on the given sequences'''

        for i in range(0, len(lnest)):
            ipos1 = i
            ipos2 = from_seq.index(to_seq[i])
            if ipos1 != ipos2:
                tmp = from_seq[ipos1]
                from_seq[ipos1] = from_seq[ipos2]
                from_seq[ipos2] = tmp
                loop1 = lnest[ipos1]
                loop2 = lnest[ipos2]
                self.__interchangeTwoLoops(loop1, loop2)
                
    #-------------------------------------------------

    def transform(self):
        '''
        Permute/interchange loops, returns a new AST for the statement being 
        transformed.
        '''

        # copy the statement
        transformed_stmt = self.stmt.replicate()

        # if only one loop is identified
        if len(self.seq) <= 1:
            return transformed_stmt

        # check if all the loop permutation indexes exist
        self.checkPermutSeq(transformed_stmt)

        # collect all permutable lists
        permut_nests, inames_seq = self.__getPermutNests(transformed_stmt)

        # try to permute for each permutable loop nest
        permut_executed = False
        for nest, inames in zip(permut_nests, inames_seq):
            skip = False
            to_seq = []
            for i in self.seq:
                if isinstance(i, str):
                    if i not in inames:
                        skip = True
                        break
                    else:
                        to_seq.append(i)
                elif i[0] in inames:
                    to_seq.append(i[0])
            if skip:
                continue
            from_seq, lnest = list(zip(*[x for x in zip(inames, nest) if x[0] in to_seq]))
            from_seq = list(from_seq)
            lnest = list(lnest)
            self.__permuteLoops(lnest, from_seq, to_seq)
            permut_executed = True

        # check if no permutation was done
        if not permut_executed:
            err('orio.module.loop.submodule.permut.transformation:  permutation cannot be executed. please check the sequence: %s' % self.seq)
        
        # return the transformed statement
        return transformed_stmt


