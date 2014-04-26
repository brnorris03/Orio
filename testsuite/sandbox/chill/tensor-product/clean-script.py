import re
import os
import sys

f = open('rose__orio_chill_.cu')
f1 = open('rose__orio_chill_clean.cu','w')

prev = f.readline()
active = False

f1.write(prev)

count = 10

registers = []

hold = False

for line in f:

	
	split1 = re.split('[\(\) ]',prev)
	split2 = re.split('(for \()',line)

	#print split2
	if len(split1) >= 3 and len(split2) >=2 :
		if split1[2] == 'cudaFree' and split2[1] == 'for (':
			#print line
			active = True
			count = 1

	split1 = re.split('(\n)',prev)
	split2 = re.split('(__global__)',line)

	if len(split1) >= 2 and len(split2) >= 2:
		if split1[1] == '\n' and split2[1] == '__global__':
			active = False

	if active == False:

		if count == 1:
			f1.write('\n}\n\n')
		addLine = True
		varLine = re.split('(double *|devO|devI|Ptr)',line)

		if len(varLine) >= 3:
			if varLine[1] == 'double ' and varLine[2] == '*':
				addLine = False
		if addLine == True:

			newLine1 = line

			if len(sys.argv) > 0:
				op = newLine1.replace(" ","")
				op1 = re.split('(\[|\(|\]|]\)|)',op)

				for i in sys.argv:
					if i == op1[0] and op1[1] == '[' and i != '0':
						op2 = re.split('('+i+'|=|\]\+)',op)
						newline = '   new_reg = (new_reg +'+op2[len(op2)-1] + '\n'
							
						registers.append(newline)
						newWrite = ''
						if sys.argv[len(sys.argv)-1] == '0':
							newWrite = '  double new_reg = 0.0;\n'
						else:
							newWrite = '  double new_reg = '+op2[1]+op2[2]+';\n'
						newWrite = newWrite + holded
						newLine1 = newWrite + newline
						newLine1 = newLine1 + '  ' + op2[1]+op2[2]+' = new_reg;\n'
						hold = False

			
			split3 = re.split('(for \()',line)

			if len(split3) >2:
				if split3[1] == 'for (':
					holded = line
					hold = True

			if hold == False:
				f1.write(newLine1)
		count = count + 1

		

	prev = line

f.close()
f1.close()

cmd = 'cp rose__orio_chill_clean.cu rose__orio_chill_.cu'
os.system(cmd)
cmd = 'rm rose__orio_chill_clean.cu'
os.system(cmd)

f = open('rose__orio_chill_.cu')

cudaMalloc = []
cudaMemcpy = []
cudaFree = []
kernelRun = []
dimGrid = []
kern = 1

move = False

for d in registers:
	print d

for line in f:

	split1 = re.split('[\(\)   ]',line)
	split2 = re.split('(<<<|>>>)',line)
	split3 = re.split('(dim3)',line)
	
	if len(split1) >= 2:
		if split1[2] == 'cudaMalloc':

			if move == True:
				move = False
				kern = kern+1
	
			temp = []
			temp.append(line)
			temp.append(kern)
			cudaMalloc.append(temp)
	
	if len(split1) >= 2:
		if split1[2] == 'cudaMemcpy':
			temp = []
			temp.append(line)
			temp.append(kern)

			cudaMemcpy.append(temp)

	if len(split2) >= 4:
		if split2[1] == '<<<' and split2[3] == '>>>':
			temp = []
			temp.append(line)
			temp.append(kern)

			kernelRun.append(temp)
			move = True
			
	if len(split3) >= 2:
		if split3[1] == 'dim3':
			temp = []
			temp.append(line)
			temp.append(kern)

			dimGrid.append(temp)
			move = True
	


f.close()



IVars = []
Iused = []
IVarsC = []


OVars = []
Oused = []


for d in range(len(cudaMemcpy)):

	op = re.split('(\(|,|\))',cudaMemcpy[d][0])
	
	if op[12] == 'cudaMemcpyHostToDevice':

		dVar = op[2]
		hVar = op[4]
		used = False
		usedVal = dVar
		for v in Iused:
			if v[0] == hVar:
				used = True
				usedVal = v[1]

		if used == False:

			space = len(Iused)

			t1 = []
			sp = re.split('(devI|Ptr)',dVar)
			sp[2] = str(space+1)
			dVarTemp = ''
			for p in sp:
				dVarTemp = dVarTemp + p

			t1.append(hVar)
			t1.append(dVarTemp)
			Iused.append(t1)
			usedVal = dVarTemp

		temp = []
		temp.append(hVar)
		temp.append(cudaMemcpy[d][1])
		temp.append(usedVal)
		temp.append(dVar)
		IVars.append(temp)

		op[2] = usedVal

	
	if op[12] == 'cudaMemcpyDeviceToHost':

		dVar = op[4]
		hVar = op[2]
		used = False
		usedVal = dVar
		for v in Iused:
			if v[0] == hVar:
				used = True
				usedVal = v[1]

		if used == False:

			space = len(Oused)

			t1 = []
			sp = re.split('(devO|Ptr)',dVar)
			sp[2] = str(space+1)
			dVarTemp = ''
			for p in sp:
				dVarTemp = dVarTemp + p

			t1.append(hVar)
			t1.append(dVarTemp)
			Oused.append(t1)
			usedVal = dVarTemp

		temp = []
		temp.append(hVar)
		temp.append(cudaMemcpy[d][1])
		temp.append(usedVal)
		temp.append(dVar)
		OVars.append(temp)

		op[4] = usedVal

	newCp = ''
	for i in op:
		newCp = newCp + i
	
	cudaMemcpy[d][0] = newCp

for d in cudaMemcpy:

	acum = 0

	counters = []
	for i in cudaMemcpy:
		if i[0] == d[0] and i[1] != d[1]:
			counters.append(acum)
		acum = acum+1
	
	
	for i in counters:
		del cudaMemcpy[i]

for d in cudaMemcpy:
	print d

for d in range(len(cudaMalloc)):

	split1 = re.split('(\(|\)|&| )',cudaMalloc[d][0])
	dVar = split1[18]

	io = re.split('(devO|devI|Ptr)',dVar)


	if io[1] == 'devO':

		for di in OVars:
			if cudaMalloc[d][1] == di[1] and dVar == di[3]:
				split1[18] = di[2]


	if io[1] == 'devI':

		for di in IVars:
			if cudaMalloc[d][1] == di[1] and dVar == di[3]:
				split1[18] = di[2]

	newMalloc = ''

	for i in split1:
		newMalloc = newMalloc + i
	
	cudaMalloc[d][0] = newMalloc


for d in cudaMalloc:

	acum = 0

	counters = []
	for i in cudaMalloc:
		if i[0] == d[0] and i[1] != d[1]:
			counters.append(acum)
		acum = acum+1
	
	
	for i in counters:
		del cudaMalloc[i]
	

for d in cudaMalloc:
	print d

for d in range(len(kernelRun)):

	kern = re.split('(\(|,|\))',kernelRun[d][0])
	
	newkern = ''
	for i in range(4,len(kern)-2):
		if kern[i] != ',':
			io = re.split('(devO|devI|Ptr)',kern[i])

			changed = False
			if io[1] == 'devI':
				for pos in IVars:
					if kern[i] == pos[3] and kernelRun[d][1] == pos[1] and changed == False:
						kern[i] = pos[2]
						changed = True
			if io[1] == 'devO':

				for pos in OVars:
					if kern[i] == pos[3] and kernelRun[d][1] == pos[1] and changed == False:
						kern[i] = pos[2]
						changed = True


	for i in kern:
		newkern = newkern + str(i)

	kernelRun[d][0] = newkern

for d in kernelRun:
	print d

VarsAdd = []
for d in Iused:
	freeC = 'cudaFree(' + d[1] + ');'
	varA = 'double *'+d[1]+';'
	VarsAdd.append(varA)
	cudaFree.append(freeC)

for d in Oused:
	freeC = 'cudaFree(' + d[1] + ');'
	varA = 'double *'+d[1]+';'
	VarsAdd.append(varA)
	cudaFree.append(freeC)

for d in cudaFree:
	print d





f = open('rose__orio_chill_.cu')
f1 = open('rose__orio_chill_clean.cu','w')

prev = f.readline()
active = False
addInfo = 0
addBrace = 10

f1.write(prev)

for line in f:

	
	split1 = re.split('[\(|\) ]',line)
	split2 = re.split('(;)',prev)

	#print split2
	if len(split1) >= 3 and len(split2) >=2 :
		if split1[2] == 'cudaMalloc' and split2[1] == ';':
			#print line
			active = True
			addInfo = addInfo+1
			addBrace = 1

	if active == True and addInfo == 1:


		for d in VarsAdd:
			f1.write('  '+d+'\n')
		for d in cudaMalloc:
			f1.write(d[0])
		for d in cudaMemcpy:
			op = re.split('(\(|,|\))',d[0])
			if op[12] == 'cudaMemcpyHostToDevice':
				f1.write(d[0])
		for d in dimGrid:
			f1.write(d[0])
		for d in kernelRun:
			f1.write(d[0])
		for d in cudaMemcpy:
			op = re.split('(\(|,|\))',d[0])
			if op[12] == 'cudaMemcpyDeviceToHost':
				f1.write(d[0])
		for d in cudaFree:
			f1.write('  ' + d + '\n')
		

	split1 = re.split('(\n)',prev)
	split2 = re.split('(__global__)',line)

	

	if len(split1) >= 2 and len(split2) >= 2:
		if split1[1] == '\n' and split2[1] == '__global__':
			active = False


	if active == False:

		if addBrace == 1:
			f1.write('\n}\n\n')
		f1.write(line)
		addBrace = addBrace +1

	prev = line

f.close()
f1.close()

cmd = 'cp rose__orio_chill_clean.cu rose__orio_chill_.cu'
os.system(cmd)
cmd = 'rm rose__orio_chill_clean.cu'
os.system(cmd)



