#                                                                                      
# Contain the transformation procedure                                                 
#                                                                                      

import sys
from orio.main.util.globals import *
import orio.module.loop.ast_lib.forloop_lib
import orio.main.util.globals as g
from orio.module.loop.ast import *

#-----------------------------------------                                             

class Transformation:
    '''Code transformation'''

    def __init__(self, protocol, msgsize,communication,stmt):
        '''To instantiate a code transformation object'''

        self.protocol = protocol
        self.msgsize = msgsize
        self.communication = communication
        self.stmt = stmt
        self.language = Globals().language
        self.flib = orio.module.loop.ast_lib.forloop_lib.ForLoopLib()

    # frequently used constants                                                       
        self.cs = {
            'int0':     NumLitExp(0, NumLitExp.INT),
            'int1':     NumLitExp(1, NumLitExp.INT),
            'int2':     NumLitExp(2, NumLitExp.INT),
            
            'lneighbor'    : ArrayRefExp(IdentExp('neighbor'), NumLitExp(0, NumLitExp.INT)),
            'rneighbor'    : ArrayRefExp(IdentExp('neighbor'), NumLitExp(1, NumLitExp.INT)),
            'commneighbor' : ArrayRefExp(IdentExp('neighbor'), IdentExp('i')),
            'commneighborj' : ArrayRefExp(IdentExp('neighbor'), IdentExp('j')),
            'maxmsg'   : IdentExp(self.msgsize), 
            'loopiters'   : IdentExp('loopiters'), 
            'dtype'    : IdentExp('datatype'),
            'id'       : IdentExp('myid'),
            'commworld': IdentExp('MPI_COMM_WORLD'),
            'tag'      : IdentExp('msgtag'),
            'i'        : IdentExp('i'),
            'j'        : IdentExp('j'),
            'ncomm'    : IdentExp('maxnumcomm'),
            'nprocs'   : IdentExp('numprocs'),
            'rindex'   : IdentExp('rindex'),
            'sindex'   : IdentExp('sindex'),
            'neighbor' : IdentExp('neighbor'),
            'nneighbour' : IdentExp('numneighbour') 
            }

      # transformation results/components                                             
  
        self.newstmts = {
            'decls'        :          [],
            'setnumneighbors' :       [],
            'setneighbors' :          [],
            'comm'         :          [],
            'wait'         :          []
            }


    #----------------------------------------------------------     

    def transform(self):
        '''To overlap communication and computation performed in the for loop'''

        numcomm = 'maxnumcomm'
        requestrecv = 'requestr'
        requestsend = 'requests'

        decls = []
        setnumneighbors = []
        setneighbors = []
        comm = []
        wait = []

        # variable declarations
        decls = [VarDeclInit('int', self.cs['tag'], self.cs['int0'])] + [VarDecl('MPI_Request ', [requestrecv + '[' + numcomm + ']'])] + [VarDecl('int',['neighbor' + '[' + numcomm + ']'])] + [VarDecl('MPI_Request ', [requestsend + '[' + numcomm + ']'])] + [VarDecl('int',['sindex'])] + [VarDecl('int',['rindex'])] + [VarDecl('MPI_Status', ['statuss' + '[' + numcomm + ']'])] + [VarDecl('MPI_Status', ['statusr' + '[' + numcomm + ']'])] + [VarDecl('int', ['numneighbour'])]

        self.newstmts['decls'] = decls

        # need to set neighbors correctly!

        if self.communication  == 1:
            setnumneighbors = [ExpStmt(BinOpExp(self.cs['nneighbour'],self.cs['int1'],BinOpExp.EQ_ASGN))]  
            setneighbors = [IfStmt(BinOpExp(BinOpExp(self.cs['nprocs'], self.cs['int2'], BinOpExp.MOD), self.cs['int0'], BinOpExp.EQ), 
                      CompStmt([ExpStmt(BinOpExp(self.cs['nneighbour'],self.cs['int1'], BinOpExp.EQ_ASGN)),              
                      IfStmt(BinOpExp(self.cs['id'], self.cs['int0'], BinOpExp.EQ),
                          ExpStmt(BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.ADD), BinOpExp.EQ_ASGN)), 
                       IfStmt(BinOpExp(self.cs['id'], BinOpExp(self.cs['nprocs'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ), ExpStmt(BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ_ASGN)),

                        IfStmt(BinOpExp(BinOpExp(self.cs['id'], self.cs['int2'], BinOpExp.MOD), self.cs['int0'], BinOpExp.EQ), ExpStmt(BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.ADD), BinOpExp.EQ_ASGN)), ExpStmt(BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ_ASGN)))))]),
                         IfStmt(BinOpExp(self.cs['id'], self.cs['int0'], BinOpExp.EQ), CompStmt([ExpStmt(BinOpExp(self.cs['nneighbour'], self.cs['int2'], BinOpExp.EQ_ASGN)), ExpStmt(BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['nprocs'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ_ASGN)), ExpStmt(BinOpExp(self.cs['rneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.ADD), BinOpExp.EQ_ASGN))]),

                           IfStmt(BinOpExp(self.cs['id'], BinOpExp(self.cs['nprocs'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ), ExpStmt(BinOpExp(self.cs['lneighbor'], self.cs['int0'], BinOpExp.EQ_ASGN)),
                               IfStmt(BinOpExp(BinOpExp(self.cs['id'], self.cs['int2'], BinOpExp.MOD), self.cs['int0'], BinOpExp.EQ), ExpStmt(BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.ADD), BinOpExp.EQ_ASGN)), ExpStmt( BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ_ASGN))))))]         
            
        elif self.communication == 2:
            setnumneighbors = [ExpStmt(BinOpExp(self.cs['nneighbour'],self.cs['int2'],BinOpExp.EQ_ASGN))]  
            setneighbors = [IfStmt(BinOpExp(self.cs['id'], self.cs['int0'], BinOpExp.EQ), CompStmt([ExpStmt(BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['nprocs'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ_ASGN)), ExpStmt(BinOpExp(self.cs['rneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.ADD), BinOpExp.EQ_ASGN))]),
            IfStmt(BinOpExp(self.cs['id'], BinOpExp(self.cs['nprocs'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ), 
               CompStmt([ExpStmt(BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ_ASGN)), ExpStmt(BinOpExp(self.cs['rneighbor'], self.cs['int0'], BinOpExp.EQ_ASGN))]), 
               CompStmt([ExpStmt(BinOpExp(self.cs['lneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.SUB), BinOpExp.EQ_ASGN)), ExpStmt(BinOpExp(self.cs['rneighbor'], BinOpExp(self.cs['id'], self.cs['int1'], BinOpExp.ADD), BinOpExp.EQ_ASGN))])))]

        elif self.communication == 3:
            setnumneighbors = [IfStmt(BinOpExp(BinOpExp(self.cs['nprocs'], self.cs['int2'], BinOpExp.MOD), self.cs['int0'], BinOpExp.EQ), 
                ExpStmt(BinOpExp(self.cs['nneighbour'], BinOpExp(self.cs['nprocs'], self.cs['int2'], BinOpExp.DIV), BinOpExp.EQ_ASGN)), 
                IfStmt(BinOpExp(BinOpExp(self.cs['id'], self.cs['int2'], BinOpExp.MOD), self.cs['int0'], BinOpExp.EQ), 
                     ExpStmt(BinOpExp(self.cs['nneighbour'], BinOpExp(self.cs['nprocs'], self.cs['int2'], BinOpExp.DIV), BinOpExp.EQ_ASGN)),
                     ExpStmt(BinOpExp(self.cs['nneighbour'], BinOpExp(BinOpExp(self.cs['nprocs'], self.cs['int2'], BinOpExp.DIV), self.cs['int1'], BinOpExp.ADD), BinOpExp.EQ_ASGN))))]

            setneighbors = [IfStmt(BinOpExp(BinOpExp(self.cs['id'], self.cs['int2'], BinOpExp.MOD), self.cs['int0'], BinOpExp.EQ),
                    CompStmt([ExpStmt(BinOpExp(self.cs['j'], self.cs['int0'], BinOpExp.EQ_ASGN)), 
                    ForStmt(BinOpExp(self.cs['i'], self.cs['int1'], BinOpExp.EQ_ASGN),                           BinOpExp(self.cs['i'], self.cs['nprocs'], BinOpExp.LT),                               BinOpExp(self.cs['i'], BinOpExp(self.cs['i'], self.cs['int2'], BinOpExp.ADD), BinOpExp.EQ_ASGN),
                            CompStmt([ExpStmt(BinOpExp(self.cs['commneighborj'], self.cs['i'], BinOpExp.EQ_ASGN)), ExpStmt(UnaryExp(self.cs['j'], UnaryExp.POST_INC))]))]),
                    CompStmt([ExpStmt(BinOpExp(self.cs['j'], self.cs['int0'], BinOpExp.EQ_ASGN)),
                    ForStmt(BinOpExp(self.cs['i'], self.cs['int0'], BinOpExp.EQ_ASGN),                           BinOpExp(self.cs['i'], self.cs['nprocs'], BinOpExp.LT),                               BinOpExp(self.cs['i'], BinOpExp(self.cs['i'], self.cs['int2'], BinOpExp.ADD), BinOpExp.EQ_ASGN),
                            CompStmt([ExpStmt(BinOpExp(self.cs['commneighborj'], self.cs['i'], BinOpExp.EQ_ASGN)), ExpStmt(UnaryExp(self.cs['j'], UnaryExp.POST_INC))]))]))]
                                 
        self.newstmts['setnumneighbors'] = setnumneighbors    

        self.newstmts['setneighbors'] = setneighbors

        #nonblocking communication 
        requestr = UnaryExp(ArrayRefExp(IdentExp('requestr'), IdentExp('i')),UnaryExp.ADDRESSOF)

        requests = UnaryExp(ArrayRefExp(IdentExp('requests'), IdentExp('i')),UnaryExp.ADDRESSOF)

        if self.protocol == 'rendezvous':
            comm = [ForStmt(BinOpExp(self.cs['i'], self.cs['int0'],BinOpExp.EQ_ASGN),
                           BinOpExp(self.cs['i'], self.cs['nneighbour'], BinOpExp.LT),
                           UnaryExp(self.cs['i'], UnaryExp.POST_INC),
                           CompStmt([ExpStmt(FunCallExp(IdentExp('MPI_Irecv'), [ArrayRefExp(IdentExp('recvbuff'),IdentExp('i')), self.cs['maxmsg'], self.cs['dtype'], self.cs['commneighbor'], self.cs['tag'], self.cs['commworld'], requestr])),
                                    ExpStmt(FunCallExp(IdentExp('MPI_Isend'), [ArrayRefExp(IdentExp('sendbuff'),IdentExp('i')), self.cs['maxmsg'], self.cs['dtype'], self.cs['commneighbor'], self.cs['tag'], self.cs['commworld'], requests]))]))]
        elif self.protocol == 'eager':
            recvr = [ForStmt(BinOpExp(self.cs['i'], self.cs['int0'],BinOpExp.EQ_ASGN),
                           BinOpExp(self.cs['i'], self.cs['nneighbour'], BinOpExp.LT),
                           UnaryExp(self.cs['i'], UnaryExp.POST_INC),
                           ExpStmt(FunCallExp(IdentExp('MPI_Irecv'), [ArrayRefExp(IdentExp('recvbuff'),IdentExp('i')), self.cs['maxmsg'], self.cs['dtype'], self.cs['commneighbor'], self.cs['tag'], self.cs['commworld'], requestr])))]

            barr =  [ExpStmt(FunCallExp(IdentExp('MPI_Barrier'),[self.cs['commworld']]))]

            sendr = [ForStmt(BinOpExp(self.cs['i'], self.cs['int0'],BinOpExp.EQ_ASGN),
                           BinOpExp(self.cs['i'], self.cs['nneighbour'], BinOpExp.LT),
                           UnaryExp(self.cs['i'], UnaryExp.POST_INC),
                           ExpStmt(FunCallExp(IdentExp('MPI_Irsend'), [ArrayRefExp(IdentExp('sendbuff'),IdentExp('i')), self.cs['maxmsg'], self.cs['dtype'], self.cs['commneighbor'], self.cs['tag'], self.cs['commworld'], requests])))]


            comm = recvr + barr + sendr

        self.newstmts['comm'] = comm

        #MPI Wait
        wait = [ForStmt(BinOpExp(self.cs['i'], self.cs['int0'],BinOpExp.EQ_ASGN),
                           BinOpExp(self.cs['i'], self.cs['nneighbour'], BinOpExp.LT),
                           UnaryExp(self.cs['i'], UnaryExp.POST_INC),
                           CompStmt([ExpStmt(FunCallExp(IdentExp('MPI_Waitany'), [self.cs['nneighbour'], IdentExp('requestr'), UnaryExp(self.cs['rindex'],UnaryExp.ADDRESSOF), IdentExp('statusr')])),
                                    ExpStmt(FunCallExp(IdentExp('MPI_Waitany'), [self.cs['nneighbour'], IdentExp('requests'), UnaryExp(self.cs['sindex'],UnaryExp.ADDRESSOF), IdentExp('statuss')]))]))]

        self.newstmts['wait'] = wait

        # create the transformed statement                                             
        if isinstance(self.stmt, orio.module.loop.ast.CompStmt):
            stmts = self.stmt.stmts
        else:
            stmts = [self.stmt]

        transformed_stmt = orio.module.loop.ast.CompStmt(
                           self.newstmts['decls'] +
                           self.newstmts['setnumneighbors'] +
                           self.newstmts['setneighbors'] +
[ForStmt(BinOpExp(self.cs['i'], self.cs['int0'], BinOpExp.EQ_ASGN), 
                            BinOpExp(self.cs['i'], self.cs['loopiters'],BinOpExp.LE),
                            UnaryExp(self.cs['i'], UnaryExp.POST_INC),
                            CompStmt(
                           self.newstmts['comm'] +
                           stmts +
                           self.newstmts['wait']))])
        
         
        # return the transformed statement                                             
        return transformed_stmt
