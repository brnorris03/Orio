#
# This orio.module.does not perform any significant code transformations at all.
# The purpose of this orio.module.is merely to provide a simple example to application developers
# for how to extend Orio with a new program transformation orio.module.
#

import sys
import orio.module.loop.submodule.submodule
from orio.module.loop.submodule.mpioverlap import transformation
from orio.main.util.globals import *

#-----------------------------------------

class MPIOverlap(orio.module.loop.submodule.submodule.SubModule):
    '''A simple rewriting module.'''

    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='C'):

        '''To instantiate a simple rewriting module.'''

        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args,stmt, language)

    #-----------------------------------------------------------------              

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''
        
        # all expected argument names
        PROTOCOL = 'protocol'
        MSG = 'msgsize'
        COMM = 'communication'

        # all expected transformation arguments
        protocol = None
        msgsize = None
        communication = None

        #iterate over all transformation arguments
        
        for aname, rhs,line_no in transf_args:
            
            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception as e:
                err('orio.module.loop.submodule.mpioverlap: %s: failed to evaluate the argument expression: %s\n --> %s: %s' % (line_no, rhs,e.__class__.__name__, e))
    
            # mpi protocol
            if aname == PROTOCOL:
                protocol = (rhs, line_no)

            # message size
            elif aname == MSG:
                msgsize = (rhs,line_no)

            # communication pattern    
            elif aname == COMM:
                communication = (rhs,line_no)

            # unknown argument name
                
            else:
                err('orio.module.loop.submodule.mpioverlap: %s: unrecognized transformation argument: "%s"' % (line_no, aname))
                

        # check for undefined transformation arguments

        if protocol == None:
            err('orio.module.loop.submodule.mpioverlap.mpioverlap: %s: missing mpi protocol argument' % self.__class__.__name__)

        if msgsize == None:
            err('orio.module.loop.submodule.mpioverlap.mpioverlap: %s: missing message size argument' % self.__class__.__name__)

        if communication == None:
            err('orio.module.loop.submodule.mpioverlap.mpioverlap: %s: missing communication argument' % self.__class__.__name__)
       
        # check semantics of the transformation arguments
        protocol,msgsize,communication = self.checkTransfArgs(protocol,msgsize, communication)

        # return information about the transformation arguments
        return (protocol,msgsize, communication)


    def checkTransfArgs(self,protocol,msgsize,communication):
        '''Check the semantics of the given transformation arguments'''

       # evaluate the protocol - add
        rhs, line_no = protocol
        if not isinstance(rhs, str):
            err('orio.module.loop.submodule.mpioverlap: %s: protocol must be a string: %s' % (line_no, rhs))
        protocol = rhs

       # evaluate the message size
        rhs, line_no = msgsize
        if not isinstance(rhs, int) or rhs <= 0:
            err('orio.module.loop.submodule.mpioverlap: %s: message size must be a positive integer: %s' % (line_no, rhs))
        msgsize = rhs

        # evaluate communication pattern
        rhs,line_no = communication
        if not isinstance(rhs,int) or rhs <=0 or rhs > 3:
            err('orio.module.loop.submodule.mpioverlap: %s: communication no. must be a positive and less than 3: %s' % (line_no, rhs))
        communication = rhs    

        # return information about the transformation arguments
        
        return (protocol, msgsize,communication)
     

    #---------------------------------------------------------------------    

    def mpiOverlap(self, protocol, msgsize, communication, stmt):
        '''To apply MPI overlap transformation'''

        debug('orio.module.loop.submodule.mpioverlap.MPIOverlap: starting mpiOverlap')

        # perform the MPI overlap transformation                                    
        t = transformation.Transformation(protocol, msgsize, communication, stmt)
        transformed_stmt = t.transform()

        # return the transformed statement                                             
        return transformed_stmt


    #---------------------------------------------------------------------    

    def transform(self):
        '''To perform code transformations'''

        # read all transformation arguments                                            
        protocol, msgsize,communication = self.readTransfArgs(self.perf_params, self.transf_args)

        # perform the mpi overlap transformation
        transformed_stmt = self.mpiOverlap(protocol, msgsize, communication, self.stmt)

        # return the transformed statement
        return transformed_stmt

