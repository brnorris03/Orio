#
# This Orio 
#

import ann_parser, orio.module.module
import sys, re, os, glob
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
	func = getFunction()
	func1 = func.replace("{",";")
	func2 = func1.replace("double *","")
	funcName = getFuncName()
	self.input_params = getInputParams()
	self.input_vars = getInputVars()
	

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
				funcs.append(funcName[1] + '_GPU_'+recipeTest[1])
				cudaVal = 'cudaize('+ recipeTest[1] + ',\"'+funcName[1] + '_GPU_' + recipeTest[1] + '\",{' 
				for key,val in self.input_vars.items():
					cudaVal = cudaVal + key + '=' + val + ','
				cudaVal = cudaVal[:-1] + '},{' + recipeTest[2] + ',' + recipeTest[3] 
				if len(recipeTest) < 6:
					cudaVal = cudaVal + ',' +recipeTest[4] +'},{})\n'

				elif len(recipeTest)  == 6:
					cudaVal = cudaVal + ',' +recipeTest[4] +',' + recipeTest[5] + '},{})\n'
				else:
					cudaVal = cudaVal + ',' +recipeTest[4] +',' + recipeTest[5] + ',' + recipeTest[6] + '},{})\n'
				cudaize = cudaize + cudaVal
				cudaCount = cudaCount+1
				
			if recipeTest[0] == 'registers':
				regs = 'copy_to_registers(' + recipeTest[1] + ',' + recipeTest[2] + ',' + stms[int(recipeTest[1])]['var'] + ')\n'
				afterCuda = afterCuda + regs

			if recipeTest[0] == 'unroll':
				regs = 'unroll(' + recipeTest[1] + ','
				index = 1
				amount = ''
				if recipeTest[3].isdigit():
					amount = recipeTest[3]
				else:
					amount = str(self.perf_params[recipeTest[3]])

				for i in stms[int(recipeTest[1])]['loops']:
					if i == recipeTest[2]:
						break
					else:
						index = index+1
				regs = regs + str(index) + ',' + amount + ')\n'
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

	
	Tblocks = []
	Mallocs = []
	dataCopy = []
	cudaFunc = []
	countBlocks = 0

	countDataDec = 0
	countDatacopy=0

	dim3 = []
	dMalloc = []
	dCopy = []
	for line in fnew2:
		funcC = filter(None,re.split('<<<|>>>| |\(|\)',line))

		if funcC[0] == 'dim3':
			dim3.append(line)
			countBlocks = countBlocks + 1
			if countBlocks == 2:
				Tblocks.append(dim3)
				dim3 = []
				countBlocks = 0

		if funcC[0] == 'cudaMemcpy':
			dCopy.append(line)
			countDatacopy = countDatacopy + 1
			if countDatacopy == 4:
				dataCopy.append(dCopy)
				dCopy = []
				countDatacopy = 0

		if funcC[0] == 'cudaMalloc':
			dMalloc.append(line)
			countDataDec = countDataDec + 1
			if countDataDec == 3:
				Mallocs.append(dMalloc)
				dMalloc = []
				countDataDec = 0

		for i in funcs:
			if funcC[0] == i:
				cudaFunc.append(line)



	variable = {}
	acum = 0
	incount = 1
	outcount = 1
	usedVal = []
	for sec in dataCopy:
		change = []
		acum2 = 0
		for j in sec:
			info2 = {}
			
			varChange = []
			inter = ''
			splitC = filter(None,re.split(',|\(|\)',j))
			dev = filter(None,re.split('(dev)',splitC[2]))
			if dev[0] != 'dev':
				if splitC[2] not in variable:
					info2['original']=splitC[1]

					dev2 = filter(None,re.split('(dev|Ptr|I|O)',splitC[1]))
					if splitC[1] not in usedVal:

						info2['new']=splitC[1]
						usedVal.append(splitC[1])
						if dev2[1] == 'I':
							incount = incount +1
						elif dev2[1] == 'O':
							outcount = outcount +1

					else:

						if dev2[1] == 'I':
							dev2[2] = str(incount)
							incount = incount+1
						elif dev2[1] == 'O':
							dev2[2] = str(outcount)
							outcount = outcount+1

						newVal = ''
						for i in dev2:
							newVal = newVal + i

						info2['new'] = newVal
						usedVal.append(newVal)
					 
					variable[splitC[2]] = info2
				inter = splitC[1]
				varChange.append(inter)
				varChange.append(variable[splitC[2]]['new'])
				change.append(varChange)

				splitD = filter(None,re.split('(&|,|\(|\))',Mallocs[acum][acum2]))
		

				newMalloc = ''
				acum3 = 0
				for i in splitD:
					if i != varChange[0]:
						newMalloc = newMalloc + i
					else:
						newMalloc = newMalloc + varChange[1]

				Mallocs[acum][acum2] = newMalloc


			elif dev[0] == 'dev':
				inter = splitC[2]
				varChange.append(inter)
				varChange.append(variable[splitC[1]]['new'])

			splitC = filter(None,re.split('(,|\(|\))',j))
			newCopy = ''
			for i in splitC:

				if i != varChange[0]:
					newCopy = newCopy + i
				else:
					newCopy = newCopy + varChange[1]

			dataCopy[acum][acum2] = newCopy
			acum2 = acum2 + 1

		newFunc = ''
		splitC = filter(None,re.split('(,|\(|\))',cudaFunc[acum]))
		acum3 = 0
		acum4 = 0

		for i in splitC:
			if i != change[acum3][0]:
				newFunc = newFunc + i
			else:
				newFunc = newFunc + change[acum3][1]
				if acum3 < len(change)-1:
					acum3 = acum3 + 1
		cudaFunc[acum] = newFunc

		acum = acum+1

	fnew2.close()
 	fnameNew = open('rose__orio_chill_clean.cu','w')
	fnew2 = open('rose__orio_chill_.cu')

	timevalStamp = 0

	fnameNew.write("#include <sys/time.h>\n")
	fnameNew.write("#include <iostream>\n")
	fnameNew.write("#include <fstream>\n")

	fnameNew.write('#include \"_orio_chill_.h\"\n\n')

	lock1 = 0
	inspector = 0
	inspFunc = 0
	varLock=0
	acumFor = 0
	newSTM = ''
	varPos = ''
	acumVar = 0
	usedVal = []
	for line in fnew2:
		check1 = filter(None,re.split('\n| ',line))
		insp = filter(None,re.split('\n| |\(|\)|,|_|;',line))

		for i in insp:
			if i == 'inspector':
				inspFunc = 1
	
		if lock1 == 0 and inspFunc == 0:
			
			fnameNew.write(line)

		inspFunc = 0
		if len(check1)==1:
			if check1[0] == '{' and lock1 ==0:
				lock1 =1

			if check1[0] == '}' and lock1 == 2:
				lock1 = 3

		if lock1 ==1:

			frees = ''
			for key,val in variable.items():
				fnameNew.write("  double *" +val['new'] +";\n")
				frees = frees + '  cudaFree('+val['new']+');\n'


			fnameNew.write('  struct timeval time1,time2;\n')
			fnameNew.write('  double time;\n')
			fnameNew.write("  std::ofstream timefile;\n")

			for i in Mallocs:
				for j in i:
					if j not in usedVal:
						fnameNew.write(j)
						usedVal.append(j)

			for i in dataCopy:
				for j in range(len(i)-1):
					if i[j] not in usedVal:
						fnameNew.write(i[j])
						usedVal.append(i[j])

			kernAcum =0
			for i in range(len(cudaFunc)):
				fnameNew.write(Tblocks[i][0])
				fnameNew.write(Tblocks[i][1])
				fnameNew.write("  gettimeofday(&time1, 0);\n")
				fnameNew.write(cudaFunc[i])
				fnameNew.write("  cudaThreadSynchronize();\n")
				fnameNew.write("  gettimeofday(&time2, 0);\n")
				fnameNew.write("  time = (1000000.0*(time2.tv_sec-time1.tv_sec) + time2.tv_usec-time1.tv_usec)/1000000.0;\n")
				fnameNew.write("  timefile.open(\"./times/time_of_"+tag+".txt\", std::ofstream::out | std::ofstream::app );\n")
				fnameNew.write('  timefile<<\"Time spent in kernel '+str(kernAcum)+': \"<<time<<std::endl;\n')
				fnameNew.write("  timefile.close();\n")

				kernAcum = kernAcum +1

			for i in dataCopy:

				fnameNew.write(i[len(i)-1])

			fnameNew.write(frees)
			if unrollFix == 1:
				fnameNew.write('}\n')
			lock1=2

		for i in insp:
			if i == 'inspector':
				if insp[len(insp)-1] != ';':
					inspector = 1


		newVar = filter(None,re.split('(newVariable|\n)| |;',line))
		
		if len(newVar) > 1:
			if newVar[0] == 'newVariable' and newVar[3] == 'newVariable' and inspector == 0:
				varPos = newVar[1]
				if acumVar > 0:
					newSTM = newSTM + '\t\t\t'
					for i in range(acumFor):
						newSTM = newSTM + '\t'
				
				for i in range(5,len(newVar)):
					newSTM = newSTM + ' ' + newVar[i]

				acumVar = acumVar+1
				varLock = 1
		
			if newVar[0] != 'newVariable' and varLock == 1:
				
				newVar2 = ''
				for i in range(acumFor):
					newVar2 = newVar2 + '\t'
				newVar2 = '  ' + newVar2 +  'newVariable' + varPos + ' = newVariable' + varPos + newSTM[:-2] +';\n'
				fnameNew.write(newVar2)
				acumVar = 0
				newSTM = ''
				varPos = ''
				varLock = 0
				acumFor = 0


		if len(newVar) > 1:
			if newVar[0] == 'for' and inspector == 0:
				acumFor = acumFor+1

		if line == '}\n' and inspector == 1:
			inspector = 0
		elif lock1==3 and inspector == 0 and varLock == 0:
			fnameNew.write(line)



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

