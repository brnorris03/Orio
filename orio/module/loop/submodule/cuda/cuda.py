#
# Loop transformation submodule that implements CUDA kernel generation
#

import os, ast
import orio.module.loop.submodule.submodule
import orio.main.util.globals as g
from orio.module.loop.submodule.cuda import transformation

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
        fprintf( fp, "'totalGlobalMem',%lld\n", 5636292608 );
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
dev_props = None

class CUDA(orio.module.loop.submodule.submodule.SubModule):
    '''The cuda transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='cuda', tinfo=None):
        '''To instantiate the transformation submodule.'''
        
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)
        self.tinfo = tinfo
        self.props = None

        
    #------------------------------------------------------------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # expected argument names
        THREADCOUNT = 'threadCount'
        BLOCKCOUNT  = 'blockCount'
        CB          = 'cacheBlocks'
        PHM         = 'pinHostMem'
        STREAMCOUNT = 'streamCount'
        DOMAIN      = 'domain'
        DOD         = 'dataOnDevice'
        UIF         = 'unrollInner'
        PREFERL1SZ  = 'preferL1Size'

        # default argument values
        szwarp  = self.props['warpSize']
        smcount = self.props['multiProcessorCount']
        threadCount  = szwarp
        blockCount   = smcount
        cacheBlocks  = False
        pinHost      = False
        streamCount  = 1
        domain       = None
        dataOnDevice = False
        unrollInner  = None
        preferL1Size = 0

        # iterate over all transformation arguments
        errors = ''
        for aname, rhs, line_no in transf_args:

            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception as e:
                g.err('orio.module.loop.submodule.cuda.cuda: %s: failed to evaluate the argument expression: %s\n --> %s: %s' % (line_no, rhs,e.__class__.__name__, e))
            
            if aname == THREADCOUNT:
                if not isinstance(rhs, int) or rhs <= 0 or rhs > self.props['maxThreadsPerBlock']:
                    errors += 'line %s: threadCount must be a positive integer less than device limit of maxThreadsPerBlock of %s: %s' % (line_no, self.props['maxThreadsPerBlock'], rhs)
                elif rhs % szwarp != 0:
                    errors += 'line %s: threadCount is not a multiple of warp size of %s: %s' % (line_no, szwarp, rhs)
                else:
                    threadCount = rhs
            elif aname == BLOCKCOUNT:
                if not isinstance(rhs, int) or rhs <= 0 or rhs > self.props['maxGridSize'][0]:
                    errors += 'line %s: %s must be a positive integer less than device limit of maxGridSize[0]=%s: %s\n' % (line_no, aname, self.props['maxGridSize'][0], rhs)
                elif rhs % smcount != 0:
                    errors += 'line %s: blockCount is not a multiple of SM count of %s: %s' % (line_no, smcount, rhs)
                else:
                    blockCount = rhs
            elif aname == CB:
                if not isinstance(rhs, bool):
                    errors += 'line %s: %s must be a boolean: %s\n' % (line_no, aname, rhs)
                else:
                    cacheBlocks = rhs
            elif aname == PHM:
                if not isinstance(rhs, bool):
                    errors += 'line %s: %s must be a boolean: %s\n' % (line_no, aname, rhs)
                else:
                    pinHost = rhs
            elif aname == STREAMCOUNT:
                if not isinstance(rhs, int) or rhs <= 0:
                    errors += 'line %s: %s must be a positive integer: %s\n' % (line_no, aname, rhs)
                else:
                    if rhs > 1:
                      overlap = self.props['deviceOverlap']
                      if overlap == 0:
                        errors += '%s=%s: deviceOverlap=%s, overlap of data transfer and kernel execution is not supported\n' % (aname, rhs, overlap)
                      concs = self.props['concurrentKernels']
                      if concs == 0:
                        errors += '%s=%s: device concurrentKernels=%s, concurrent kernel execution is not supported\n' % (aname, rhs, concs)
                    streamCount = rhs
            elif aname == DOMAIN:
                if not isinstance(rhs, str):
                    errors += 'line %s: %s must be a string: %s\n' % (line_no, aname, rhs)
                else:
                    domain = rhs
            elif aname == DOD:
                if not isinstance(rhs, bool):
                    errors += 'line %s: %s must be a boolean: %s\n' % (line_no, aname, rhs)
                else:
                    dataOnDevice = rhs
            elif aname == UIF:
                if not isinstance(rhs, int) or rhs <= 0:
                    errors += 'line %s: %s must be a positive integer: %s\n' % (line_no, aname, rhs)
                else:
                    unrollInner = rhs
            elif aname == PREFERL1SZ:
                if not isinstance(rhs, int) or rhs not in [16,32,48]:
                    errors += 'line %s: %s must be either 16, 32 or 48 KB: %s\n' % (line_no, aname, rhs)
                else:
                    major = self.props['major']
                    if major < 2:
                      errors += '%s=%s: L1 cache is not resizable on compute capability less than 2.x, current comp.cap.=%s.%s\n' % (aname, rhs, major, self.props['minor'])
                    elif major < 3 and rhs == 32:
                      errors += '%s=%s: L1 cache cannot be set to %s on compute capability less than 3.x, current comp.cap.=%s.%s\n' % (aname, rhs, rhs, major, self.props['minor'])
                    preferL1Size = rhs
            else:
                g.err('%s: %s: unrecognized transformation argument: "%s"' % (self.__class__, line_no, aname))

        if not errors == '':
          raise Exception('%s: errors evaluating transformation args:\n%s' % (self.__class__, errors))

        # return evaluated transformation arguments
        return {
          THREADCOUNT:threadCount,
          BLOCKCOUNT:blockCount,
          CB:cacheBlocks,
          PHM:pinHost,
          STREAMCOUNT:streamCount,
          DOMAIN:domain,
          DOD:dataOnDevice,
          UIF:unrollInner,
          PREFERL1SZ:preferL1Size}

    #------------------------------------------------------------------------------------------------------------------

    def getDeviceProps(self):
      '''Get device properties'''

      # First, check if user specified the device properties file
      if self.tinfo.device_spec_file:
        qout = self.tinfo.device_spec_file
      else:
          # generate the query code
          qsrc  = "enum_cuda_props.cu"
          qexec = qsrc + ".o"
          qout  = qexec + ".props"
          if not os.path.exists(qout):
            # check for nvcc
            qcmd = 'which nvcc'
            status = os.system(qcmd)
            if status != 0:
              g.err("%s: could not locate nvcc with '%s'" % (self.__class__, qcmd))
    
            try:
              f = open(qsrc, 'w')
              f.write(CUDA_DEVICE_QUERY_SKELET)
              f.close()
            except:
              g.err('%s: cannot open file for writing: %s' % (self.__class__, qsrc))
            
            # compile the query
            cmd = 'nvcc -o %s %s' % (qexec, qsrc)
            status = os.system(cmd)
            if status:
              g.err('%s: failed to compile cuda device query code: "%s"' % (self.__class__, cmd))
    
            # execute the query
            runcmd = './%s' % (qexec)
            status = os.system(runcmd)
            if status:
              g.err('%s: failed to execute cuda device query code: "%s"' % (self.__class__, runcmd))
            os.remove(qsrc)
            os.remove(qexec)
        
      # read device properties
      props = {}
      try:
        f = open(qout, 'r')
        for line in f:
            eline = ast.literal_eval(line)
            props[eline[0]] = eline[1]
        f.close()
      except:
        g.err('%s: cannot open query output file for reading: %s' % (self.__class__, qout))
  
      if props['devId'] == -2:
        g.err("%s: there is no CUDA 1.0 enabled GPU on this machine" % self.__class__)
      
      if props['major'] < 2 and props['minor'] < 3:
        g.warn("%s: running on compute capability less than 1.3 is not recommended, detected %s.%s." % (self.__class__, props['major'], props['minor']))

      # set the arch to the latest supported by the device
      if self.tinfo is None:
          bcmd = "gcc"
      else:
          bcmd = self.tinfo.build_cmd
          
      if bcmd.startswith('gcc'):
        bcmd = 'nvcc'
      if bcmd.find('-arch') == -1:
        bcmd += ' -arch=sm_' + str(props['major']) + str(props['minor'])
      if self.perf_params is not None and 'CFLAGS' in self.perf_params and bcmd.find('@CFLAGS') == -1:
        bcmd += ' @CFLAGS'
      if self.tinfo is not None:
        self.tinfo.build_cmd = bcmd

      # return queried device props
      return props

    #------------------------------------------------------------------------------------------------------------------

    def cudify(self, stmt, targs):
        '''Apply CUDA transformations'''
        
        g.debug('orio.module.loop.submodule.cuda.CUDA: starting CUDA transformations')

        # perform transformation
        if self.props is None:
            self.props = self.getDeviceProps()
        t = transformation.Transformation(stmt, self.props, targs, self.tinfo)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt

    #------------------------------------------------------------------------------------------------------------------

    def transform(self):
        '''The implementation of the abstract transform method for CUDA'''

        # read device properties
        global dev_props # initialize device properties only once
        if dev_props is None:
            dev_props = self.getDeviceProps()
        if self.props is None:
            self.props = dev_props
        
        # read all transformation arguments
        targs = self.readTransfArgs(self.perf_params, self.transf_args)
        
        # perform the transformation of the statement
        transformed_stmt = self.cudify(self.stmt, targs)
        
        return transformed_stmt


