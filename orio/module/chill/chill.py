#
# This Orio
#

import ann_parser, orio.module.module
import sys, re, os, glob
import collections
from orio.main.util.globals import *

#-----------------------------------------

class CHiLL(orio.module.module.Module):
    '''Orio's interface to the CHiLL source transformation infrastructure. '''

    def __init__(self, perf_params, module_body_code, annot_body_code, line_no, indent_size, language='C',tinfo=None):
        '''To instantiate the CHiLL rewriting module.'''

        orio.module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)
        ##Module.__init__(self, perf_params, module_body_code, annot_body_code,
        ##                              line_no, indent_size, language)
        self.tinfo = tinfo

    #---------------------------------------------------------------------

    def transform(self):
        '''To simply rewrite the annotated code'''

        # to create a comment containing information about the class attributes
        comment = '''
        /*
         perf_params = %s
         module_body_code = "%s"
         annot_body_code = "%s"
         line_no = %s
         indent_size = %s
        */
        ''' % (self.perf_params, self.module_body_code, self.annot_body_code, self.line_no, self.indent_size)


        # CHiLL annotations are of the form:
        # /*@ begin Chill ( transform Recipe(recipe filename) ) @*/
        # ...
        # The code to be transformed by CHiLL here
        # ...
        # /*@ end Chill @*/
        code = self.annot_body_code
        fname = '_orio_chill_.c'
        hname = '_orio_chill_.h'
        func = Globals().getFuncDecl()
        funcName = Globals().getFuncName()
        self.input_params = Globals().getFuncInputParams()
        self.input_vars = Globals().getFuncInputVars()


        #print "Informatio variables: \nperf_params: ",self.perf_params,"\nmodule_body_code: ",self.module_body_code,"\nline_no: ",self.line_no,"\nindent_size: ",self.indent_size

        defines = ''


        cudaize = []
        permute = []
        registers = []
        distribute = ''

        output_code = ''

        cont = True

        exits = 1

        stm = 0

        for key, val in list(self.perf_params.items()):
            if 'TX' in key:
                stm +=1

        TB = []
        for i in range(stm):


            temp = [self.perf_params['TX' + str(i)],self.perf_params['TY' + str(i)],self.perf_params['BX' + str(i)],self.perf_params['BY' + str(i)]]

            if self.perf_params['TX' + str(i)] == '1':
                cont = False
                exits = 0

            if len(temp) != len(set(temp)):
                cont = False
                exits = 0

            else:

                TB.append(temp)

        if cont == True:

            counter = 0
            loopsNumber = []

            unrolls = []

            codeInfo = [_f for _f in re.split('\n',code) if _f]

            temp = []
            for line in codeInfo:
                if 'for' in line and 'dummyLoop' not in line:
                    temp.append(line)

                elif len(temp) > 0:

                    loopsNumber.append(temp)
                    temp = []

            stamp = ''
            for key,val in list(self.perf_params.items()):

                stamp += '_'+str(val)


            annot = [_f for _f in re.split('\t|\n| ',self.module_body_code) if _f]
            stm = 0
            dist = ''
            for line in annot:

                lineInfo = [_f for _f in re.split('\(|\)',line) if _f]

                if lineInfo[0] == 'cuda':

                    cuda = 'cudaize('+str(stm)+',"'+funcName+'_GPU_'+ str(stm)+'\",{'
                    for vars1 in self.input_vars:
                        cuda += vars1[0] + '=' + vars1[1] + ','

                    cuda = cuda[:-1] + '},{block={\"' + TB[stm][2] + '\",\"' + TB[stm][3] + '\"},thread={\"' + TB[stm][0] + '\",\"' + TB[stm][1] + '\"}},{})'
                    stm+=1
                    cudaize.append(cuda)

                if lineInfo[0] == 'registers':
                    registers.append(line)

                if lineInfo[0] == 'unroll':
                    unrollInfo = [_f for _f in re.split(',|\"',lineInfo[1]) if _f]
                    unrollPos = unrollInfo[1]+'++'
                    stmPos = int(unrollInfo[0])
                    loopPos = 0
                    for l in range(len(loopsNumber[stmPos])):
                        loopsInfo = [_f for _f in re.split(' |\(|\)|;',loopsNumber[stmPos][l]) if _f]
                        if unrollPos == loopsInfo[3]:

                            loopPos = l

                    ammount = 1
                    if unrollInfo[2].isdigit():
                        ammout = unrollInfo[2]
                    else:
                        ammount = self.perf_params[unrollInfo[2]]



                    unrollCmd = 'unroll('+unrollInfo[0] + ','+ str(loopPos+2) + ',' + str(ammount)+')'

                    unrolls.append(unrollCmd)

                if lineInfo[0] == 'distribute':
                    dist = lineInfo[1]


            distribute = ''
            if dist != '':

                distribute = 'distribute({'
                for i in range(stm):
                    distribute += str(i) + ','
                distribute = distribute[:-1] + '},' + dist + ')'

            rname = 'recipe'+stamp+'.lua'


            recipe = 'init(\"'+fname+'\",\"'+funcName+'\",0)\ndofile(\"cudaize.lua\")\n\n'

            defines = ''
            for inputs in self.input_params:
                defines += '#define ' + inputs[0] + ' ' + inputs[1] + '\n'
                recipe += inputs[0] + '=' + inputs[1] + '\n'

            recipe += distribute + '\n\n'

            for cuda in cudaize:
                recipe += cuda.replace(',\"1\"}','}') + '\n'

            recipe += '\n'
            for unroll in unrolls:

                recipe += unroll + '\n'

            code = defines + '\n' + func + '\n{' + code + '\n}\n'


            cfile = open(fname,'w')
            cfile.write(code)
            cfile.close()

            hfile = open(hname,'w')
            hfile.write(func + ';\n')
            hfile.close()


            rfile = open(rname,'w')
            rfile.write(recipe)
            rfile.close()

            os.system('cuda-chill '+rname)

            if not os.path.isdir('./recipes'):
                try:
                    os.system('mkdir recipes')
                    os.system('mkdir outputs')
                    os.system('mkdir times')
                except:
                    err('orio.module.chill.chill:  failed to create folders')

            try:
                os.system('mv ' + rname + ' recipes/./')

            except:
                err('orio.module.chill.chill:  failed to move files')


            chillout = 'rose__orio_chill_.cu'


            fname = open(chillout)

            newHost = ''

            dCopy = []
            dMalloc = []
            dGrid = []
            cudaKern = []
            pointers = {}

            lock = 0
            lock2 = 1


            lockHost = 0

            cudaKernels = []
            kernel = ''
            for line in fname:

                ###################Grab the mallocs, memcpy, etc###################

                lineInfo = [_f for _f in re.split(' |(double|cudaMalloc|cudaMemcpy|dim|cudaFree|<<<|__global__|;)',line) if _f]

                if 'void '+funcName+'(' in line:
                    lockHost = 1

                if lineInfo[0] == 'cudaMemcpy':
                    dCopy.append(line)

                if lineInfo[0] == 'cudaMalloc':
                    dMalloc.append(line)

                if lineInfo[0] == 'dim':
                    dGrid.append(line)

                if len(lineInfo) > 1:
                    if lineInfo[1] == '<<<':
                        cudaKern.append(line)

                if lock == 0:
                    newHost += line

                if line == '{\n':
                    lock = 1


        #################################Grab the kernels for future use #################################
                if lineInfo[0] == '__global__' and lineInfo[len(lineInfo)-2] != ';':
                    lock2 = 0

                if lock2 == 0:
                    kernel += line

                if line == '}\n':
                    lock2 = 1
                    if kernel != '':
                        cudaKernels.append(kernel)
                        kernel = ''


            timeInfo = '#include <sys/time.h>\n'
            timeInfo += '#include <iostream>\n'
            timeInfo += '#include <fstream>\n'
            timeInfo += '#include \"_orio_chill_.h\"\n\n'

            newHost = timeInfo + newHost + '\n'

            fname.close()

            dMalloc = []
            cudaFrees = []

            counterIn = 1
            counterOut = 1
            stm = 0
            stm2 = 0
            for i in range(len(dCopy)):

                if 'cudaMemcpyHostToDevice' in dCopy[i]:
                    if stm2 == 1:
                        stm+=1
                        stm2 = 0

                    cpInfo = [_f for _f in re.split(',| |\(|\)',dCopy[i]) if _f]

                    tempKern = [_f for _f in re.split('(>>>)',cudaKern[stm]) if _f]

                    if cpInfo[2] in pointers:
                        devPointer = pointers[cpInfo[2]]

                    else:
                        devPointer = 'devI' + str(counterIn) + 'Ptr'
                        pointers[cpInfo[2]] = devPointer
                        counterIn += 1
                        cudaFrees.append('cudaFree('+devPointer+')')


                    tempKern[2] = tempKern[2].replace(cpInfo[1],devPointer,1)
                    cudaKern[stm] = tempKern[0] + tempKern[1] + tempKern[2]
                    dCopy[i] = dCopy[i].replace(cpInfo[1],devPointer)

                    mallocs = 'cudaMalloc(((void **)(&'+devPointer+')),'+cpInfo[3]+' * sizeof(double ));'

                    dMalloc.append(mallocs)


                if 'cudaMemcpyDeviceToHost' in dCopy[i]:
                    cpInfo = [_f for _f in re.split(',| |\(|\)',dCopy[i]) if _f]
                    tempKern = [_f for _f in re.split('(>>>)',cudaKern[stm]) if _f]

                    if cpInfo[1] in pointers:
                        devPointer = pointers[cpInfo[1]]

                    else:
                        devPointer = 'devO' + str(counterOut) + 'Ptr'
                        pointers[cpInfo[1]] = devPointer
                        counterOut += 1
                        cudaFrees.append('cudaFree('+devPointer+')')

                    tempKern[2] = tempKern[2].replace(cpInfo[2],devPointer,1)
                    cudaKern[stm] = tempKern[0] + tempKern[1] + tempKern[2]
                    dCopy[i] = dCopy[i].replace(cpInfo[2],devPointer)

                    mallocs = 'cudaMalloc(((void **)(&'+devPointer+')),'+cpInfo[3]+' * sizeof(double ));'

                    dMalloc.append(mallocs)


                    stm2 = 1

            dCopy = list(set(dCopy))
            dMalloc = list(set(dMalloc))


            for key, vals in list(pointers.items()):

                newHost += '  double *' + vals + ';\n'


            newHost += '\n'

            newHost += '  struct timeval time1, time2;\n  double time;\n  std::ofstream timefile;\n'
            for mallocs in dMalloc:
                newHost += '  ' + mallocs + '\n'

            copyIns = ''
            copyOuts = ''
            for copies in dCopy:
                if 'cudaMemcpyDeviceToHost' in copies:
                    copyOuts += copies
                if 'cudaMemcpyHostToDevice' in copies:
                    copyIns += copies

            newHost += '\n' + copyIns + '\n'

            for grids in dGrid:
                newHost += grids

            newHost += '\n'

            newHost += '  gettimeofday(&time1, 0);\n'
            for cuda in cudaKern:
                newHost += cuda

            newHost += '  cudaThreadSynchronize();\n'
            newHost += '  gettimeofday(&time2, 0);\n'
            newHost += '  time = (1000000.0*(time2.tv_sec-time1.tv_sec) + time2.tv_usec-time1.tv_usec)/1000000.0;\n'
            newHost += '  timefile.open(\"./times/time_'+stamp+'.txt\", std::ofstream::out | std::ofstream::app );\n'
            newHost += '  timefile<<\"Time spent in rose__orio_chill_ '+stamp+'.cu: \"<<time<<std::endl;\n'
            newHost += '  timefile.close();\n\n'

            newHost += '\n' + copyOuts + '\n'

            for frees in cudaFrees:

                newHost += '  ' + frees + ';\n'

            newHost += '\n}\n\n'


            stm = 0
            for cuda in cudaKernels:

                lock = 0
                fors = []

                lineInfo = [_f for _f in re.split('\n',cuda) if _f]
                StmLine = []
                var = 1

                for info in lineInfo:
                    if 'for' in info:
                        lock = 1
                        fors.append(info)
                    if lock == 1 and 'for' not in info:
                        StmLine.append(info)
                    if lock == 0:
                        newHost += info + '\n'

                newHost += '  double newVar' + str(var) + ';\n'
                regs = [_f for _f in re.split('\(|\)|registers|,|\"',registers[stm]) if _f]

                copyReg = [_f for _f in re.split('=',StmLine[0]) if _f]
                copyReg[0] = copyReg[0].strip()

                StmLine[0] = StmLine[0].replace(copyReg[0],'newVar' + str(var))
                StmLine[0] = StmLine[0].replace(';','')



                for i in range(1,len(StmLine)):

                    StmLine[i] = StmLine[i].replace('= ','')
                    StmLine[i] = StmLine[i].replace(copyReg[0],'')
                    StmLine[i] = StmLine[i].replace(';','')





                counter = 0
                temp = 0
                temp2 = ''
                space = '  '
                closing = ''
                for loops in fors:

                    if regs[1] + ' +=' in loops:
                        newHost += space + 'newVar' + str(var) + ' = '+ copyReg[0] + ';\n'
                        par = ''
                        closing = space +  copyReg[0] + ' = newVar' + str(var) +';\n' + closing

                    loops2 = loops.replace('{','')
                    newHost += loops2 + '{\n'

                    closing = space + '}\n' + closing

                    space += '  '



                for stms in StmLine:
                    if '}' not in stms:
                        newHost += stms + '\n'
                newHost =newHost[:-1] + ';\n' + closing + '\n}\n'


                stm += 1





            newCode = open('rose__orio_chill_.cu','w')

            newCode.write(newHost)
            newCode.close()


            cmd = 'nvcc -O3 -arch=sm_20 rose__orio_chill_.cu -c -o rose__orio_chill_.o'

            try:
                os.system(cmd)
                cmd = 'mv rose__orio_chill_.cu outputs/rose__orio_chill_' + stamp + '.cu'
                os.system(cmd)
            except:
                err('orio.module.chill.chill:  failed to compile with nvcc: %s' % cmd)


            output_code = '\n\n //Testing with: recipe' + stamp + '.lua'
            output_code += '\n //Output file is: rose__orio_chill_' +stamp + '.cu'


            output_code = output_code + '\n\n#include \"_orio_chill_.h\" \n\n'

            func2 = func.replace('void','')
            func2 = func2.replace('double *','')

            output_code += func2 + ';\n'


            return output_code
#               exits = 1

        exit()
#       else:

#               code2 = func + '{\n}'
#               cfile = open(fname,'w')
#               cfile.write(code2)
#               cfile.close()

#               hfile = open(hname,'w')
#               hfile.write(func + ';\n')
#               hfile.close()

#               output_code = '\n\n //Testing bad run'
#               output_code += '\n //don\'t use for measurments'


 #              output_code = output_code + '\n\n#include \"_orio_chill_.h\" \n\n'

#               func2 = func.replace('void','')
#               func2 = func2.replace('double *','')

#               output_code += func2+ ';\n'


        return output_code
