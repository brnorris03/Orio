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
	func1 = func.replace("{",";")
	func2 = func1.replace("double *","")
	funcName = Globals().getFuncName()
	self.input_params = Globals().getFuncInputParams()
	self.input_vars = Globals().getFuncInputVars()
	

	#print "Informatio variables: \nperf_params: ",self.perf_params,"\nmodule_body_code: ",self.module_body_code,"\nline_no: ",self.line_no,"\nindent_size: ",self.indent_size


	transCMD = re.split(r'[\n\t]+',self.module_body_code)
	##print transCMD,len(transCMD)
	tInfo = re.split(r'[ ()\t]+',transCMD[0])
	##print tInfo
	del tInfo[-1]

	cmd = ''
	scriptCMD = ""
	CU = 1
	recipeFound = False
	cname = ''
	if tInfo[1] == 'Recipe':	##NEED FIX
		if len(transCMD) > 1:
			err('orio.module.chill.chill: Recipe file can\'t be added when other transformations are included')
		elif len(tInfo) == 3:
			cname = tInfo[2] 
			cname1, ctype = os.path.splitext(cname)
			recipeFound = True
			if ctype != '.lua':
				err('orio.module.chill.chill: Wrong file type')
		
		elif len(tInfo) < 3:
			err('orio.module.chill.chill: No recipe filename give')
	defines = ''
	cleanScript = False

	cleanArgv = []
	stms = []
	funcs = []
	cudaCount = 0
	unrollFix = 0
	if recipeFound == False:
		scriptCMD=''
		transforms=''
		cudaize=''
		afterCuda=''
		distVal = ''

		for trans in range(len(transCMD)):
			stmInfo = {}
			recipeTest = filter(None,re.split('\t|\n|\(|\)|:|,',transCMD[trans]))

			if recipeTest[0] == 'stm':
			
				stmInfo['num'] = recipeTest[1]
				loops = []
				for i in range(2,len(recipeTest)-1):
					loops.append(recipeTest[i])

				stmInfo['loops'] = loops
				stmInfo['var'] = recipeTest[len(recipeTest)-1]
				stms.append(stmInfo)
				

			if recipeTest[0] == 'distribute':
				distr = recipeTest[0] + '({' 

				for i in range(len(stms)):
					distr = distr + str(i) + ','
				distr = distr[:-1] + '},' + recipeTest[1] + ')\n'
				distVal = recipeTest[1]
				transforms = transforms + distr

			if recipeTest[0] == 'tile':
				controlL = recipeTest[len(recipeTest)-1]
				amount = ''
				if recipeTest[3].isdigit():
					amount = recipeTest[3]
				else:
					amount = str(self.perf_params[recipeTest[3]])

				tiles = 'tile_by_index(' + recipeTest[1] + ',{' + recipeTest[2] + '},{' + amount + '},{l1_control=' +controlL + '},{'
				newLoop = []
				for i in stms[int(recipeTest[1])]['loops']:
					if i == recipeTest[2]:
						newLoop.append(controlL)
						tiles = tiles + controlL + ','
					newLoop.append(i)
					tiles = tiles + i + ','
				stms[int(recipeTest[1])]['loops'] = newLoop
				tiles = tiles[:-1] + '})\n'

				transforms = transforms + tiles
			if recipeTest[0] == 'permute':
				perm = 'tile_by_index(' + recipeTest[1] + ',{},{},{},{'
				newLoop = []
				for i in range(2,len(recipeTest)):
					newLoop.append(recipeTest[i])
					perm = perm + recipeTest[i] + ','
				stms[int(recipeTest[1])]['loops'] = newLoop
				perm= perm[:-1] + '})\n'
	
				transforms = transforms + perm

			if recipeTest[0] == 'cuda':
				TBt = filter(None,re.split('\t|\n|\(|\)|:|\{|\}|=|block|thread',transCMD[trans]))
				blocksT = filter(None,re.split(',',TBt[2]))
				threadsT = filter(None,re.split(',',TBt[4]))

				blocksNew = 'block={'
				blocksNew1 = 'block={'


				Bt = []
				Bt1 = []
				Tt1 = []

				for i in range(len(blocksT)):

					if blocksT[i] in self.perf_params.keys():
						if self.perf_params[blocksT[i]] != "1":
							blocksNew = blocksNew + '\"' + self.perf_params[blocksT[i]]  + '\",'
						blocksNew1 = blocksNew1 + '\"' + self.perf_params[blocksT[i]]  + '\",'

						Bt.append('\"' + self.perf_params[blocksT[i]] + '\"')

					else:
						blocksNew = blocksNew +  blocksT[i] +','
						Bt.append(blocksT[i])


				blocksNew = blocksNew[:-1] + '}'
				blocksNew1 = blocksNew1[:-1] + '}'

				threadsNew = 'thread={'

				threadsNew1 = 'thread={'

				for i in range(len(threadsT)):

					if threadsT[i] in self.perf_params.keys():
						if self.perf_params[threadsT[i]] != "1":
							threadsNew = threadsNew + '\"' + self.perf_params[threadsT[i]] + '\",'
						threadsNew1 = threadsNew1 + '\"' + self.perf_params[threadsT[i]] + '\",'

						Bt.append('\"' +self.perf_params[threadsT[i]]+'\"')
					else:
						threadsNew = threadsNew +  threadsT[i] + ','
						threadsNew1 = threadsNew1 +  threadsT[i] + ','

						Bt.append(threadsT[i])
				threadsNew = threadsNew[:-1] + '}'
				threadsNew1 = threadsNew1[:-1] + '}'

				inter =  [x for x, y in collections.Counter(Bt).items() if y > 1]



				if len(inter) > 0:
					print "Warning: Threand and block decomposition wrong. Version not counted."
					print blocksNew1
					print threadsNew1 
					return output_code


				funcs.append(funcName[1] + '_GPU_'+recipeTest[1])
				cudaVal = 'cudaize('+ recipeTest[1] + ',\"'+funcName[1] + '_GPU_' + recipeTest[1] + '\",{' 
				for key,val in self.input_vars.items():
					cudaVal = cudaVal + key + '=' + val + ','
				cudaVal = cudaVal[:-1] + '},{' 
				cudaVal = cudaVal + blocksNew + ',' + threadsNew + '},{})\n'
				cudaize = cudaize + cudaVal
				cudaCount = cudaCount+1
				
			if recipeTest[0] == 'registers':
				regs = 'copy_to_registers(' + recipeTest[1] + ',' + recipeTest[2] + ',' + stms[int(recipeTest[1])]['var'] + ')\n'
				afterCuda = afterCuda + regs

			if recipeTest[0] == 'fuse':
				fusion = recipeTest[0] + '({'
				lenfuse = (len(recipeTest)-1) * 3
				for i in range(lenfuse):
					fusion = fusion + str(i) + ','
				fusion = fusion[:-1]
				fusion = fusion + '},' + distVal + ')\n'
				afterCuda = afterCuda + fusion


			if recipeTest[0] == 'unroll':
				regs = 'unroll(' + recipeTest[1] + ','
				amount = ''
				if recipeTest[3].isdigit():
					amount = recipeTest[3]
				else:
					amount = str(self.perf_params[recipeTest[3]])

				if int(amount) > 1: 
					index = 0
					for i in stms[int(recipeTest[1])]['loops']:
						if i == recipeTest[2]:
							break
						index = index + 1
					regs = regs + str(index) + ',' + amount + ')\n'
					if self.perf_params[recipeTest[3]] != 1:
						unrollFix = 1
					afterCuda = afterCuda + regs

			

		scriptCMD = transforms + cudaize + afterCuda

		if len(self.input_params) > 0:
			for key,value in self.input_params.items():
				defines = defines + '#define ' + key + ' '+ value + '\n'

		tag = ''
		for key,value in self.perf_params.iteritems():
			tag = tag + "_"+str(value)
	

		cname = 'recipe'+tag+'.lua'
	
		try:
		    cfile = open(cname,'w')
		    cfile.write("init(\""+fname.replace("'","") + "\",\""+funcName[1]+"\",0)\n") 
		    cfile.write("dofile(\"cudaize.lua\")\n\n")
		    for key,value in self.input_params.items():
			    cfile.write(key + '='+value + '\n')
		    
		    cfile.write('\n' + scriptCMD + '\n')
		    cfile.close()

		except:
		    err('orio.module.chill.chill: cannot open file for writing: %s' % cname)
		##copy the recipes to another file to avoid a mess up in file
		if not os.path.exists('recipes'):
    			os.makedirs('recipes')

		if not os.path.exists('times'):
    			os.makedirs('times')

	
		cmd = 'cp '+cname+' recipes/./'
		try:
		    os.system(cmd)
		except:
	            err('orio.module.chill.chill:  failed to run command: %s' % cmd)

	if not os.path.isfile(fname):
		try:
		    f = open(fname,'w')
		    f.write(defines + '\n')
		    f.write(funcName[0] + ' ' + funcName[1] + '(')
		    f.write(func)
		    f.write('){')
		    f.write(code)
		    f.write("\n}\n\n")
		    f.close()



		except:
		    err('orio.module.chill.chill: cannot open file for writing: %s' % fname)

	if not os.path.isfile(hname):
		try:
		    h = open(hname,'w')    
		    h.write(funcName[0] + ' ' + funcName[1] + '(' + func + ');\n')
		    h.write("\n\n")
		    h.close()

		except:
		    err('orio.module.chill.chill: cannot open file for writing: %s' % hname)

	

	#RUN CUDA-CHILL MUTE FOR DEBUG PURPOUSE
	try:

	    os.path.isfile(cname)

	except:
	    err('orio.module.chill.chill: cannot open file recipe for CUDA-CHiLL: %s' % cname)


	cmd = 'cuda-chill %s' % (cname)
	info('orio.module.chill.chill: running CUDA-CHiLL with command: %s' % cmd)

	try:
	    os.system(cmd)
	except:
            err('orio.module.chill.chill:  failed to run command: %s' % cmd)

		


#################################re-arrange for better performance######################



	fnew2 = open('rose__orio_chill_.cu')

	dataCopy = {}
	dataMalloc = {}
	dataFree = []
	cudaKern = {}
	variables = []
	Grid = {}

	vari = 0
	kernCall = 0
	kernelName = 'kernel_' + str(kernCall)
	acumData = 0

	dCopy = []
	dMalloc = []
	dGrid = []

	for line in fnew2:
	
		lineInfo = filter(None,re.split(' |(double|cudaMalloc|cudaMemcpy|dim|cudaFree|\_GPU\_)',line))
	
		if lineInfo[0] == 'cudaMemcpy':
			dCopy.append(line)

		if lineInfo[0] == 'cudaMalloc':
			dMalloc.append(line)
			acumData = acumData + 1

		if lineInfo[0] == 'dim':
			dGrid.append(line)

		if len(lineInfo) > 2:
			if lineInfo[1] == '_GPU_':
				cudaKern[kernelName] = line


		if lineInfo[0] == 'cudaFree':
			acumData = acumData - 1

		if acumData == 0 and len(dMalloc) > 0:

			dataCopy[kernelName] = dCopy
			dCopy = []

			dataMalloc[kernelName] = dMalloc
			dMalloc = []
			Grid[kernelName] = dGrid
			dGrid = []

			kernCall = kernCall + 1
			kernelName = 'kernel_' + str(kernCall)



	usedValues = {}
	newCopyIn = []
	newCopyOut = []
	newMalloc = []
	newKernel = [None] * len(cudaKern)

	dataCounter = [1,1]
	for key in dataCopy:
		oldValue = {}
		for j in dataCopy[key]:
			lineInfo = filter(None, re.split('(\(|,|\))',j))
			pointer = filter(None,re.split('(dev|I|O|Ptr)',lineInfo[2]))

			if pointer[0] == 'dev':
				oldValue[lineInfo[2]] = lineInfo[4]

				val = 1
				if pointer[1] == 'I':
					val = 0


				if not lineInfo[4] in usedValues:
					usedValues[lineInfo[4]] = pointer[0] + pointer[1] + str(dataCounter[val]) + pointer[3]

				dataCounter[val] = dataCounter[val] + 1

			if lineInfo[4] in usedValues:
				lineInfo[2] = usedValues[lineInfo[4]]

			if lineInfo[2] in usedValues:
				lineInfo[4] = usedValues[lineInfo[2]]

					
			cudaCopy = ''
			for i in lineInfo:
				cudaCopy = cudaCopy + i

			if pointer[0] == 'dev':
				if not cudaCopy in newCopyIn:
					newCopyIn.append(cudaCopy)
			else:
				if not cudaCopy in newCopyOut:
					newCopyOut.append(cudaCopy)
	
		kernInfo = filter(None, re.split('(\(|,|_GPU_|<<<|\))',cudaKern[key]))
		newKern = ''
		for i in range(len(kernInfo)):
			if kernInfo[i] in oldValue:
				kernInfo[i] = usedValues[oldValue[kernInfo[i]]]
			newKern = newKern + kernInfo[i]

		newKernel[int(kernInfo[2])] = newKern
		cudaKern[key] = newKern

		for j in dataMalloc[key]:
			lineInfo = filter(None,re.split('(\(|\&|\))',j))

			cudaMalloc = ''
			for i in range(len(lineInfo)):
				if lineInfo[i] in oldValue:
					lineInfo[i] = usedValues[oldValue[lineInfo[i]]]
				cudaMalloc = cudaMalloc+lineInfo[i]

			if not cudaMalloc in newMalloc:
				newMalloc.append(cudaMalloc)


	fnew2.close()
	

	for key in usedValues:
		dataFree.append('  cudaFree('+usedValues[key]+');\n')
		variables.append('  double *'+usedValues[key]+';\n')

 	fnameNew = open('rose__orio_chill_clean.cu','w')
	fnew2 = open('rose__orio_chill_.cu')

	timevalStamp = 0

	fnameNew.write("#include <sys/time.h>\n")
	fnameNew.write("#include <iostream>\n")
	fnameNew.write("#include <fstream>\n")

	fnameNew.write('#include \"_orio_chill_.h\"\n\n')

	inspectorCounter = 0
	lock = 0
	lockBody = 0
	kernAcum = 0
	kernLocks = 0
	forLock = 0
	acum = 0
	inspLock = 0
	unrollAcum = 0
	combineSt = {}
	for line in fnew2:

		lineInfo = filter(None,re.split('(_inspector|\(|\))',line))
		lineInfo3 = filter(None, re.split(' ',line))
		lineInfo2 = filter(None,re.split('(newVariable|for| = )',line))
	
		if len(lineInfo) > 1:
			if lineInfo[1] == '_inspector':		
				inspectorCounter = inspectorCounter + 1
				inspLock = 3

		if lineInfo3[0] == '__global__':
			combineSt = {}


		if lineInfo[0] == '}\n' and inspectorCounter == inspLock:
			lock = 0
			kernLocks = 1


		if kernLocks == 1:
			if len(lineInfo2) > 4:
				if lineInfo2[4] == 'newVariable':
					forLock = 1

		if lock == 0 and forLock == 0:
			fnameNew.write(line)

		if len(lineInfo2) > 3:
			if forLock == 1 and lineInfo2[1] == 'newVariable':
		
				
				var = lineInfo2[1] + lineInfo2[2]

			
				if var in combineSt and len(lineInfo2) >5:
					combineSt[var] = combineSt[var] + ' \t\t' + lineInfo2[5][1:]
					combineSt[var] = combineSt[var][:-2] + '\n'

				if not var in combineSt and len(lineInfo2) >5:
					
					combineSt[var] = lineInfo2[5][1:]
					combineSt[var] = combineSt[var][:-2] + '\n'
				
		if forLock == 1 and lineInfo3[0] == '}\n':
			unrollAcum = unrollAcum + 1
		if len(lineInfo2)>2:
			if forLock == 1 and lineInfo2[2] == 'newVariable':

				for key in combineSt:
					fnameNew.write('    ' + key + ' = ' + key + ' ' + combineSt[key][:-1] + ';\n')

				if unrollAcum > 0:
					fnameNew.write('    }\n')
				fnameNew.write(line)
				forLock = 0

		if lineInfo[0] == '{\n' and lockBody == 0:
			lock = 1
			lockBody = 1

		if lockBody == 1:

			for i in variables:
				fnameNew.write(i)

			fnameNew.write('  struct timeval time1, time2;\n  double time;\n  std::ofstream timefile;\n')
			for i in newMalloc:
				fnameNew.write(i)

			fnameNew.write('\n')
			for i in newCopyIn:
				fnameNew.write(i)

			for key in Grid:
				fnameNew.write('\n')
				for j in Grid[key]:
					fnameNew.write(j)

			fnameNew.write('\n')
			for i in newKernel:
				fnameNew.write('  gettimeofday(&time1, 0);\n')
				fnameNew.write(i)
				fnameNew.write("  cudaThreadSynchronize();\n")
				fnameNew.write("  gettimeofday(&time2, 0);\n")
				fnameNew.write("  time = (1000000.0*(time2.tv_sec-time1.tv_sec) + time2.tv_usec-time1.tv_usec)/1000000.0;\n")
				fnameNew.write("  timefile.open(\"./times/time_of_"+tag+".txt\", std::ofstream::out | std::ofstream::app );\n")
				fnameNew.write('  timefile<<\"Time spent in kernel '+str(kernAcum)+': \"<<time<<std::endl;\n')
				fnameNew.write("  timefile.close();\n\n")

				kernAcum = kernAcum +1


			for i in newCopyOut:
				fnameNew.write(i)

			fnameNew.write('\n')
			for i in dataFree:
				fnameNew.write(i) 

			lockBody = lockBody + 1
				
	fnameNew.close()

##########################################################################
	cmd = 'cp rose__orio_chill_clean.cu rose__orio_chill_.cu'
	try:
		os.system(cmd)
	except:
	     err('orio.module.chill.chill:  failed to run command: %s' % cmd)

	cmd = 'rm rose__orio_chill_clean.cu'

	try:
		os.system(cmd)
	except:
	     err('orio.module.chill.chill:  failed to run command: %s' % cmd)


	cmd = 'nvcc -O3 -arch=sm_20 rose__orio_chill_.cu -c -o rose__orio_chill_.o'
	
	try:
		os.system(cmd)
	except:
		err('orio.module.chill.chill:  failed to compile with nvcc: %s' % cmd)


	fname2 = ''
	if recipeFound == False:
		fname1, ctype = os.path.splitext(fname)
		fname2 = 'rose_' + fname1 + tag + ctype + 'u'
	else:
		fname2 = fname+'u'

	info('orio.module.chill.chill: Output located in: rose_%s' % fname2)

	if recipeFound == False: 
		if not os.path.exists('outputs'):
			os.makedirs('outputs')
	
		cmd = 'cp rose_'+fname+'u outputs/' + fname2
		try:
		    os.system(cmd)
		except:
			err('orio.module.chill.chill:  failed to run command: %s' % cmd)

		f1 = open("./times/time_of_"+tag+".txt","a")

		f1.write("Filename: "+fname2+"\n\n")
		f1.close()
		


	#CLEAN THE WORKING DIRECTORY
	if recipeFound == False:
		cname = 'recipe'+tag+'.lua'
		cmd = 'rm '+cname
		try:
		    os.system(cmd)
		except:
	            err('orio.module.chill.chill:  failed to run command: %s' % cmd)

		cmd = 'rm rose_'+fname+'u'
		try:
		    os.system(cmd)
		except:
	            err('orio.module.chill.chill:  failed to run command: %s' % cmd)


        # Do nothing except output the code annotated with a comment for the parameters that were specified
        # TODO: this is where we use the provided CHiLL recipe or generate a new one
        # Then invoke CHiLL to produce the output_code
	
	## We annotate which recipe file was used to generate this version
	output_code = '\n\n //Testing with file: ' + cname
	output_code = output_code + '\n //Output file is: ' +fname2


        output_code = output_code + '\n\n#include \"_orio_chill_.h\" \n\n'
	output_code = output_code + funcName[1] + '(' + func2 + ');\n'


        # return the output code
        return output_code

