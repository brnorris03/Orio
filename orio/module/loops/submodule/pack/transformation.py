from orio.module.loops.ast import *

#----------------------------------------------------------------------------------------------------------------------
class Transformation:

    def __init__(self, stmt, targs):
        self.stmt = stmt
        self.targs = targs
        self.toprefetch = targs['prefetch']
        self.prefetch_distance = targs['prefetch_distance']

    #--------------------------------------------------------------------------

    def transform(self):

        if len(self.targs) == 0:
            return self.stmt

        nv = NodeVisitor()
        getIds = lambda n: [n.name] if isinstance(n, IdentExp) else []
        #allIds = nv.collectTD(getIds, self.stmt)

        # find the prefetch ids
        # find array ids
        getArrayIds = lambda n: nv.collectTD(getIds, n.exp) if isinstance(n, ArrayRefExp) else []
        arrayIds = nv.collectTD(getArrayIds, self.stmt)
        
        # find lhs and rhs ids
        shortAssignOps = [BinOpExp.PLUSEQ, BinOpExp.MINUSEQ, BinOpExp.MULTEQ, BinOpExp.DIVEQ, BinOpExp.MODEQ,
                          BinOpExp.BSHLEQ, BinOpExp.BSHREQ, BinOpExp.BANDEQ, BinOpExp.BXOREQ, BinOpExp.BOREQ]
        assignOps = shortAssignOps + [BinOpExp.EQ]
        getLhsIds = lambda n: nv.collectTD(getIds, n.lhs) if isinstance(n, BinOpExp) and n.op_type in assignOps else []
        getRhsIds = lambda n: nv.collectTD(getIds, n.rhs) if isinstance(n, BinOpExp) and n.op_type==BinOpExp.EQ else \
                              nv.collectTD(getIds, n) if isinstance(n, BinOpExp) and n.op_type in shortAssignOps else []
        lhsIds = nv.collectTD(getLhsIds, self.stmt)
        rhsIds = nv.collectTD(getRhsIds, self.stmt)
        
        # find lhs and rhs array idents
        lhsArrayIds = set(arrayIds) & set(lhsIds)
        rhsArrayIds = set(arrayIds) & set(rhsIds)
        
        # prefetch each rhsArrayId for read access
        def addPrefetch(id):
            def rw(n):
                if isinstance(n, ExpStmt):
                    existsId = nv.collectTD(lambda a: [(True,a.sub)] if isinstance(a, ArrayRefExp) and a.exp.name==id else [], n)
                    if len(existsId) > 0:
                        for atuple in existsId:
                            atuple2=atuple[1]
                            idxId = IdentExp("");
                            if isinstance(atuple2, UnaryExp) and isinstance(atuple2.exp, IdentExp):
                                idxId = atuple2.exp
                            elif isinstance(atuple2, IdentExp):
                                idxId = atuple2
                            fetch = ExpStmt(CallExp(IdentExp('__builtin_prefetch'),
                                                    [UnaryExp(UnaryExp.ADDRESSOF, ArrayRefExp(IdentExp(id),
                                                                                              BinOpExp(BinOpExp.PLUS, idxId, IdentExp(self.prefetch_distance)))),
                                                     LitExp(LitExp.INT, 0),
                                                     LitExp(LitExp.INT, 0)]))
                        n = CompStmt([n, fetch])
                    return n
                else:
                    return n
            return rw

        for rhsArrayId in rhsArrayIds:
            transformed_stmt = nv.rewriteBU(addPrefetch(rhsArrayId), self.stmt)

        # return the transformed statement
        return transformed_stmt
#----------------------------------------------------------------------------------------------------------------------


