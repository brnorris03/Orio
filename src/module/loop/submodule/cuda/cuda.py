#
# Loop transformation submodule.that implements CUDA kernel generation
#

import os, ast
import orio.module.loop.submodule.submodule
import orio.main.util.globals as g
import transformation

#----------------------------------------------------------------------------------------------------------------------
CUDA_DEVICE_QUERY_SKELET = r'''
#include <stdio.h>
#include "cuda.h"

int main( void ) { 
    FILE *fp = fopen("enum_cuda_props.cu.o.props", "w");

    cudaDeviceProp  prop;
    int count = 0;
    cudaGetDeviceCount(&count);
    for (int i=0; i<count; i++) {
        cudaGetDeviceProperties(&prop, i); 
        fprintf( fp, "'devId',%d\n", i );
        fprintf( fp, "'name','%s'\n", prop.name );
        fprintf( fp, "'major',%d\n", prop.major );
        fprintf( fp, "'minor',%d\n", prop.minor );
        fprintf( fp, "'clockRate',%d\n", prop.clockRate );
        fprintf( fp, "'deviceOverlap',%d\n", prop.deviceOverlap );
        fprintf( fp, "'kernelExecTimeoutEnabled',%d\n", prop.kernelExecTimeoutEnabled );
        fprintf( fp, "'totalGlobalMem',%ld\n", prop.totalGlobalMem );
        fprintf( fp, "'totalConstMem',%ld\n", prop.totalConstMem );
        fprintf( fp, "'memPitch',%ld\n", prop.memPitch );
        fprintf( fp, "'textureAlignment',%ld\n", prop.textureAlignment );

        fprintf( fp, "'multiProcessorCount',%d\n", prop.multiProcessorCount );
        fprintf( fp, "'sharedMemPerBlock',%ld\n", prop.sharedMemPerBlock );
        fprintf( fp, "'regsPerBlock',%d\n", prop.regsPerBlock );
        fprintf( fp, "'warpSize',%d\n", prop.warpSize );
        fprintf( fp, "'maxThreadsPerBlock',%d\n", prop.maxThreadsPerBlock );
        fprintf( fp, "'maxThreadsDim',(%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
        fprintf( fp, "'maxGridSize',(%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );

        fprintf( fp, "'integrated',%d\n", prop.integrated );
        fprintf( fp, "'canMapHostMemory',%d\n", prop.canMapHostMemory );
        fprintf( fp, "'computeMode',%d\n", prop.computeMode );
        fprintf( fp, "'maxTexture1D',%d\n", prop.maxTexture1D );
        fprintf( fp, "'maxTexture2D',(%d,%d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1] );
        fprintf( fp, "'maxTexture3D',(%d,%d,%d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2] );
        fprintf( fp, "'concurrentKernels',%d\n", prop.concurrentKernels );
    }
    if (count == 0) { // defaults
        fprintf( fp, "'devId',%d\n", -1);
        fprintf( fp, "'name',%s\n", "'Tesla C2070'" );
        fprintf( fp, "'major',%d\n", 2 );
        fprintf( fp, "'minor',%d\n", 0 );
        fprintf( fp, "'clockRate',%d\n", 1147000 );
        fprintf( fp, "'deviceOverlap',%d\n", 1 );
        fprintf( fp, "'kernelExecTimeoutEnabled',%d\n", 0 );
        fprintf( fp, "'totalGlobalMem',%ld\n", 5636292608 );
        fprintf( fp, "'totalConstMem',%d\n", 65536 );
        fprintf( fp, "'memPitch',%d\n", 2147483647 );
        fprintf( fp, "'textureAlignment',%d\n", 512 );

        fprintf( fp, "'multiProcessorCount',%d\n", 14 );
        fprintf( fp, "'sharedMemPerBlock',%d\n", 49152 );
        fprintf( fp, "'regsPerBlock',%d\n", 32768 );
        fprintf( fp, "'warpSize',%d\n", 32 );
        fprintf( fp, "'maxThreadsPerBlock',%d\n", 1024 );
        fprintf( fp, "'maxThreadsDim',(%d,%d,%d)\n", 1024, 1024, 64 );
        fprintf( fp, "'maxGridSize',(%d,%d,%d)\n", 65535, 65535, 65535 );

        fprintf( fp, "'integrated',%d\n", 0 );
        fprintf( fp, "'canMapHostMemory',%d\n", 1 );
        fprintf( fp, "'computeMode',%d\n", 0 );
        fprintf( fp, "'maxTexture1D',%d\n", 65536 );
        fprintf( fp, "'maxTexture2D',(%d,%d)\n", 65536, 65535 );
        fprintf( fp, "'maxTexture3D',(%d,%d,%d)\n", 2048, 2048, 2048 );
        fprintf( fp, "'concurrentKernels',%d\n", 1 );
    }
    fclose(fp);
}
'''
#----------------------------------------------------------------------------------------------------------------------

class CUDA(orio.module.loop.submodule.submodule.SubModule):
    '''The cuda transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='cuda'):
        '''To instantiate the transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)

        # TODO: future transformations here, e.g., 
        #self.cudastream_smod = orio.module.loop.submodule.cudastream.cudastream.CudaStream()
        
    #-----------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # all expected argument names
        THREADCOUNT = 'threadCount'

        # all expected transformation arguments
        threadCount = None

        # iterate over all transformation arguments
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception, e:
                g.err('orio.module.loop.submodule.cuda.cuda: %s: failed to evaluate the argument expression: %s\n --> %s: %s' % (line_no, rhs,e.__class__.__name__, e))
                
            # thread count
            if aname == THREADCOUNT:
                threadCount = (rhs, line_no)
    
            # unknown argument name
            else:
                g.err('orio.module.loop.submodule.cuda.cuda: %s: unrecognized transformation argument: "%s"' % (line_no, aname))

        # check for initialization of mandatory transformation arguments
        if threadCount == None:
            g.err('orio.module.loop.submodule.cuda.cuda: %s: missing threadCount argument' % self.__class__.__name__)

        # check semantics of the transformation arguments
        threadCount = self.checkTransfArgs(threadCount)

        # return information about the transformation arguments
        return threadCount

    #-----------------------------------------------------------------

    def checkTransfArgs(self, threadCount):
        '''Check the semantics of the given transformation arguments'''
        
        # sanity check
        rhs, line_no = threadCount
        if not isinstance(rhs, int) or rhs <= 0:
            g.err('orio.module.loop.submodule.cuda.cuda: %s: threadCount must be a positive integer: %s' % (line_no, rhs))
        threadCount = rhs

        # return information about the transformation arguments
        return threadCount

    #-----------------------------------------------------------------

    def getDeviceProps(self):
        '''Get device properties'''

        # write the query code
        qsrc  = "enum_cuda_props.cu"
        qexec = qsrc + ".o"
        qout  = qexec + ".props"
        try:
            f = open(qsrc, 'w')
            f.write(CUDA_DEVICE_QUERY_SKELET)
            f.close()
        except:
            g.err('orio.module.loop.submodule.cuda.cuda: cannot open file for writing: %s' % qsrc)
        
        # compile the query
        cmd = 'nvcc -o %s %s' % (qexec, qsrc)
        status = os.system(cmd)
        if status:
            g.err('orio.module.loop.submodule.cuda.cuda: failed to compile cuda device query code: "%s"' % cmd)

        # execute the query
        runcmd = './%s' % (qexec)
        status = os.system(runcmd)
        if status:
            g.err('orio.module.loop.submodule.cuda.cuda: failed to execute cuda device query code: "%s"' % runcmd)
        
        # read query results
        props = {}
        try:
            f = open(qout, 'r')
            for line in f:
                eline = ast.literal_eval(line)
                props[eline[0]] = eline[1]
            f.close()
        except:
            g.err('orio.module.loop.submodule.cuda.cuda: cannot open query output file for reading: %s' % qout)

        # clean up
        os.remove(qsrc)
        os.remove(qexec)
        os.remove(qout)
        
        # return queried device props
        return props

    #-----------------------------------------------------------------

    def cudify(self, stmt, threadCount, props):
        '''Apply CUDA transformations'''
        
        g.debug('orio.module.loop.submodule.cuda.CUDA: starting CUDA transformations')

        # perform transformation
        t = transformation.Transformation(stmt, threadCount, props)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt

    #-----------------------------------------------------------------

    def transform(self):
        '''The implementation of the abstract transform method for CUDA'''

        # read all transformation arguments
        threadCount = self.readTransfArgs(self.perf_params, self.transf_args)
        
        # read device properties
        props = self.getDeviceProps()
        
        # perform the transformation of the statement
        transformed_stmt = self.cudify(self.stmt, threadCount, props)
        
        return transformed_stmt
