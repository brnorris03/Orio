#!/usr/bin/env python
'''
Created on Feb 17, 2015

@author: norris
'''
import re, sys, os

besttol = 0.2
fairtol = 0.4

def readTuningLog(filename):
    f=open(filename,'r')
    lines = f.readlines()
    f.close()
    return lines

'''
    @param lines list of lines (strings)
'''
def convertToARFF(lines):
    global besttol, worsttol
    featuresre = re.compile(r'^(\[\'.*\'\])$')
    datare = re.compile(r'^.*({.*})$')
    subdatare = re.compile(r'^({.*}), "transform_time": ([\d\.]+)}$')
    datalist = []
    mintime = sys.float_info.max
    maxtime = 0
    buf = '@RELATION autotune_data\n'
    for l in lines:
        m = featuresre.match(l)
        if m:
            featureslist = eval(m.groups(1)[0])
            for feature in featureslist:
                buf += '@ATTRIBUTE %s NUMERIC\n' % feature
            buf += '@ATTRIBUTE TIME NUMERIC\n'
            buf += '@ATTRIBUTE class {best,fair,worst}'
            buf += '\n@DATA\n\n'
            #print featureslist
        m = datare.match(l)
        if m: 
            #{"T1_I": 1, "T1_J": 1, "U_J": 1, "U_I": 1, "T2_I": 1, "T2_J": 1, "U1_I": 1, 
            # "OMP": false, "VEC2": false, "VEC1": false, "RT_I": 1, "SCR": false, "RT_J": 1}, "transform_time": 1.88651704788208
            tmpdata = (m.groups(1)[0]).split('}')
            tmpdata[0] = tmpdata[0]
            paramvalues = dict((k.strip(), v.strip()) for k,v in 
              (item.split(':') for item in tmpdata[0].split(',')))
            for k,v in paramvalues.items(): 
                if type(v) == bool: 
                    if v: paramvalues[k] = 1 
                    else: paramvalues[k] = 0
            dtime = float(tmpdata[-2].split(':')[-1])
            datalist.append((paramvalues,dtime))
            if dtime < mintime: mintime = dtime
            elif dtime > maxtime: maxtime = dtime
            
    
    for d in datalist:
        params=d[0]
        tm = d[1]
        if tm <= (1.0+besttol) * mintime: label = 'best'
        elif tm >= (1.0+fairtol) * mintime: label = 'fair'
        else: label = 'worst'
        
        buf += ','.join(params.values()) + ',' + str(tm) + ',' + label + '\n'
        
    return buf

def writeToFile(buf, fname):
    with open(os.path.basename(fname) + '.arff', "w") as arff_file:
        arff_file.write(buf)
        arff_file.close()
    return 
        

if __name__ == '__main__':
    fname = sys.argv[1]
    lines = readTuningLog(fname)
    buf = convertToARFF(lines)
    writeToFile(buf,fname)
    pass