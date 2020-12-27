#
# Loop transformation submodule that implements OpenCL kernel generation
#

import os
import orio.module.loop.submodule.submodule
import orio.main.util.globals as g
from orio.module.loop import ast
from orio.module.loop.submodule.opencl import transformation

OPENCL_DEVICE_QUERY_SKELET = r'''
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#define BProp(dev, name) printBProp(dev, #name, name)
#define UIProp(dev, name) printUIProp(dev, #name, name)
#define ULProp(dev, name) printULProp(dev, #name, name)
#define SizeProp(dev, name) printSizeProp(dev, #name, name)
#define SProp(dev, name) printSProp(dev, #name, name)

void printBProp(cl_device_id dev, char * name, cl_device_info prop) {
    cl_bool boolValue;
    clGetDeviceInfo(dev, prop, sizeof(cl_bool), &boolValue, NULL);
    const char * v = boolValue ? "True" : "False";
    printf("'%s',%s\n", name, v);
}

void printUIProp(cl_device_id dev, char * name, cl_device_info prop) {
    cl_uint uintValue;
    clGetDeviceInfo(dev, prop, sizeof(cl_uint), &uintValue, NULL);
    printf("'%s',%u\n", name, uintValue);
}

void printULProp(cl_device_id dev, char * name, cl_device_info prop) {
    cl_ulong ulongValue;
    clGetDeviceInfo(dev, prop, sizeof(cl_ulong), &ulongValue, NULL);
    printf("'%s',%llu\n", name, ulongValue);
}

void printSizeProp(cl_device_id dev, char * name, cl_device_info prop) {
    size_t sizeValue;
    clGetDeviceInfo(dev, prop, sizeof(size_t), &sizeValue, NULL);
    printf("'%s',%zu\n", name, sizeValue);
}

void printSProp(cl_device_id dev, char * name, cl_device_info prop) {
    size_t valueSize;
    char * charValue;
    clGetDeviceInfo(dev, prop, 0, NULL, &valueSize);
    charValue = (char*) malloc(valueSize);
    clGetDeviceInfo(dev, prop, valueSize, charValue, NULL);
    printf("'%s','%s'\n", name, charValue);
    free(charValue);
}

int main() {

    cl_uint i, j;
    char* info;
    size_t infoSize;
    cl_uint platformCount;
    cl_platform_id *platforms;
    const char* attributeNames[5] = { "CL_PLATFORM_NAME", "CL_PLATFORM_VENDOR",
        "CL_PLATFORM_VERSION", "CL_PLATFORM_PROFILE", "CL_PLATFORM_EXTENSIONS" };
    const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
        CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
    const size_t attributeCount = sizeof(attributeNames) / sizeof(char*);
    clGetPlatformIDs(5, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    // Platforms
    for (i = 0; i < platformCount; i++) {
        printf("'PLATFORM',%d\n", i);
        for (j = 0; j < attributeCount; j++) {
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);
            printf("'%s','%s'\n", attributeNames[j], info);
            free(info);
        }

        // Devices
        cl_uint deviceCount;
        cl_device_id* devices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id *) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        cl_uint d;
        for(d = 0; d < deviceCount; d++) {
            printf("'DEVICE','%d'\n", d);

            SProp(devices[d], CL_DEVICE_NAME);
            SProp(devices[d], CL_DEVICE_VERSION);
            SProp(devices[d], CL_DEVICE_OPENCL_C_VERSION);
            SProp(devices[d], CL_DRIVER_VERSION);
            UIProp(devices[d], CL_DEVICE_ADDRESS_BITS);
            BProp(devices[d], CL_DEVICE_AVAILABLE);
            BProp(devices[d], CL_DEVICE_COMPILER_AVAILABLE);
            BProp(devices[d], CL_DEVICE_ENDIAN_LITTLE);
            BProp(devices[d], CL_DEVICE_ERROR_CORRECTION_SUPPORT);
            SProp(devices[d], CL_DEVICE_EXTENSIONS);
            ULProp(devices[d], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
            UIProp(devices[d], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
            ULProp(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE);
            BProp(devices[d], CL_DEVICE_IMAGE_SUPPORT);
            SizeProp(devices[d], CL_DEVICE_IMAGE2D_MAX_HEIGHT);
            SizeProp(devices[d], CL_DEVICE_IMAGE2D_MAX_WIDTH);
            SizeProp(devices[d], CL_DEVICE_IMAGE3D_MAX_DEPTH);
            SizeProp(devices[d], CL_DEVICE_IMAGE3D_MAX_HEIGHT);
            SizeProp(devices[d], CL_DEVICE_IMAGE3D_MAX_WIDTH);
            ULProp(devices[d], CL_DEVICE_LOCAL_MEM_SIZE);
            UIProp(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY);
            UIProp(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS);
            UIProp(devices[d], CL_DEVICE_MAX_CONSTANT_ARGS);
            ULProp(devices[d], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
            ULProp(devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            SizeProp(devices[d], CL_DEVICE_MAX_PARAMETER_SIZE);
            UIProp(devices[d], CL_DEVICE_MAX_READ_IMAGE_ARGS);
            UIProp(devices[d], CL_DEVICE_MAX_SAMPLERS);
            SizeProp(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE);
            UIProp(devices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);

            cl_uint maxDim;
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxDim, NULL);
            size_t * itemSizes = malloc(sizeof(size_t) * maxDim);
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*maxDim, itemSizes, NULL);
            printf("'CL_DEVICE_MAX_WORK_ITEM_SIZES',(");
            size_t item;
            for(item = 0; item < maxDim; ++item) {
                printf("%zu,", itemSizes[item]);
            }
            printf(")\n");
            free(itemSizes);

            UIProp(devices[d], CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
            UIProp(devices[d], CL_DEVICE_MEM_BASE_ADDR_ALIGN);
            UIProp(devices[d], CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE);
            UIProp(devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
            UIProp(devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
            UIProp(devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
            UIProp(devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
            UIProp(devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
            UIProp(devices[d], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
            SProp(devices[d], CL_DEVICE_PROFILE);
            SizeProp(devices[d], CL_DEVICE_PROFILING_TIMER_RESOLUTION);
        }

        free(devices);
    }

    free(platforms);
    return 0;

}
'''
#----------------------------------------------------------------------------------------------------------------------
dev_props = None
class OpenCL(orio.module.loop.submodule.submodule.SubModule):
    '''The OpenCL transformation submodule.'''
    
    def __init__(self, perf_params = None, transf_args = None, stmt = None, language='opencl', tinfo=None):
        '''To instantiate the transformation submodule.'''
        orio.module.loop.submodule.submodule.SubModule.__init__(self, perf_params, transf_args, stmt, language)
        self.tinfo = tinfo
        self.props = None

        
    #------------------------------------------------------------------------------------------------------------------

    def readTransfArgs(self, perf_params, transf_args):
        '''Process the given transformation arguments'''

        # expected argument names
        PLATFORM    = 'platform'
        DEVICE      = 'device'
        WORKGROUPS  = 'workGroups'
        WORKITEMS   = 'workItemsPerGroup'
        CB          = 'cacheBlocks'
        STREAMCOUNT = 'streamCount'
        UIF         = 'unrollInner'
        CLFLAGS     = 'clFlags'
        THREADCOUNT = 'threadCount'
        BLOCKCOUNT  = 'blockCount'
        VECHINT     = 'vecHint'
        SIZEHINT    = 'sizeHint'

        # default argument values
        platform = 0
        device = 0
        workGroups  = None
        workItemsPerGroup   = None
        cacheBlocks  = False
        streamCount  = 1
        unrollInner  = None
        clFlags      = None
        vecHint      = 0
        sizeHint     = False

        # iterate over all transformation arguments
        errors = ''
        for aname, rhs, line_no in transf_args:
            # evaluate the RHS expression
            try:
                rhs = eval(rhs, perf_params)
            except Exception as e:
                g.err('orio.module.loop.submodule.opencl.opencl: %s: failed to evaluate the argument expression: %s\n --> %s: %s' % (line_no, rhs,e.__class__.__name__, e))

            if aname == PLATFORM:
                # TODO: validate
                platform = rhs
            elif aname == DEVICE:
                # TODO: validate
                device = rhs
            elif aname == WORKGROUPS:
                # TODO: validate
                workGroups = rhs
            elif aname == WORKITEMS:
                # TODO: validate
                workItemsPerGroup = rhs
            elif aname == CB:
                # TODO: validate
                cacheBlocks = rhs
            elif aname == STREAMCOUNT:
                # TODO: validate
                streamCount = rhs
            elif aname == UIF:
                # TODO: validate
                unrollInner = rhs
            elif aname == CLFLAGS:
                clFlags = rhs
            elif aname == THREADCOUNT:
                g.warn("Interpreting CUDA threadCount as OpenCL workItemsPerGroup")
                workItemsPerGroup = rhs
            elif aname == BLOCKCOUNT:
                g.warn("Interpreting CUDA blockCount as OpenCL workGroups")
                workGroups = rhs
            elif aname == VECHINT:
                vecHint = rhs
            elif aname == SIZEHINT:
                sizeHint = rhs
            else:
                g.err('%s: %s: unrecognized transformation argument: "%s"' % (self.__class__, line_no, aname))

        if not errors == '':
            raise Exception('%s: errors evaluating transformation args:\n%s' % (self.__class__, errors))

        # return evaluated transformation arguments
        return {
          PLATFORM:platform,
          DEVICE:device,
          WORKGROUPS:workGroups,
          WORKITEMS:workItemsPerGroup,
          CB:cacheBlocks,
          STREAMCOUNT:streamCount,
          UIF:unrollInner,
          CLFLAGS:clFlags,
          VECHINT:vecHint,
          SIZEHINT:sizeHint,}

    #------------------------------------------------------------------------------------------------------------------

    def getDeviceProps(self):
        '''Get device properties'''
        # write the query code
        qsrc  = "enum_opencl_props.c"
        qexec = qsrc + ".o"
        qout  = qexec + ".props"

        try:
            f = open(qsrc, 'w')
            f.write(OPENCL_DEVICE_QUERY_SKELET)
            f.close()
        except:
            g.err('%s: cannot open file for writing: %s' % (self.__class__, qsrc))
        
        # compile the query
        if self.tinfo is not None and self.tinfo.build_cmd is not None:
            cmd = self.tinfo.build_cmd
        else:
            cmd = 'gcc -framework OpenCL'
            
        cmd += ' -o %s %s' % (qexec, qsrc)

        status = os.system(cmd)
        if status:
            g.err('%s: failed to compile OpenCL device query code: "%s"' % (self.__class__, cmd))

        # execute the query
        runcmd = './%s > ./%s' % (qexec, qout)
        status = os.system(runcmd)
        if status:
            g.err('%s: failed to execute OpenCL device query code: "%s"' % (self.__class__, runcmd))
        os.remove(qsrc)
        os.remove(qexec)
        
        # read device properties
        platforms = []
        try:
            f = open(qout, 'r')
            mostRecentWasDevice = False
            for line in f:
                eline = ast.literal_eval(line)
                if eline[0] == 'PLATFORM':
                    mostRecentWasDevice = False
                    platforms.append({'devices':[]})
                elif eline[0] == 'DEVICE':
                    mostRecentWasDevice = True
                    platforms[-1]['devices'].append({})
                else:
                    if mostRecentWasDevice:
                        platforms[-1]['devices'][-1][eline[0]] = eline[1]
                    else:
                        platforms[-1][eline[0]] = eline[1]
            f.close()
            #print platforms
        except:
            g.err('%s: cannot open query output file for reading: %s' % (self.__class__, qout))
            
        # return queried device props
        return platforms


    #------------------------------------------------------------------------------------------------------------------

    def openclify(self, stmt, targs):
        '''Apply OpenCL transformations'''
        
        g.debug('orio.module.loop.submodule.opencl.opencl: starting OpenCL transformations')

        # perform transformation
        t = transformation.Transformation(stmt, self.props, targs, self.tinfo)
        transformed_stmt = t.transform()

        # return the transformed statement
        return transformed_stmt

    #------------------------------------------------------------------------------------------------------------------

    def transform(self):
        '''The implementation of the abstract transform method for OpenCL'''
        # read device properties
        global dev_props # initialize device properties only once
        if dev_props is None:
            dev_props = self.getDeviceProps()
        if self.props is None:
            self.props = dev_props
            
        # read all transformation arguments
        targs = self.readTransfArgs(self.perf_params, self.transf_args)
        g.Globals().metadata.update(targs)
    
        # perform the transformation of the statement
        transformed_stmt = self.openclify(self.stmt, targs)
        
        return transformed_stmt


