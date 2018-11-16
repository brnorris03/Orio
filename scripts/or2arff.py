#!/usr/bin/env python
'''
Created on Feb 17, 2015

@author: norris
'''
import re, sys, os, argparse

def readTuningLog(filename):
    f=open(filename,'r')
    lines = f.readlines()
    f.close()
    return lines

'''
    @param lines list of lines (strings)
'''
def convertToARFF(lines,besttol,fairtol,includetimes=False):
    featuresre = re.compile(r'^(\[\'.*\'\])$')
    datare = re.compile(r'^\(run \d+\) \| ({.*}?)$')
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
            if includetimes:
                buf += '@ATTRIBUTE MINTIME NUMERIC\n'
                buf += '@ATTRIBUTE MAXTIME NUMERIC\n'
                buf += '@ATTRIBUTE AVGTIME NUMERIC\n'
            buf += '@ATTRIBUTE class {best,fair,worst}'
            buf += '\n@DATA\n\n'
            #print featureslist
        m = datare.match(l)
        if m: 
            #{"T1_I": 1, "T1_J": 1, "U_J": 1, "U_I": 1, "T2_I": 1, "T2_J": 1, "U1_I": 1, 
            # "OMP": false, "VEC2": false, "VEC1": false, "RT_I": 1, "SCR": false, "RT_J": 1}, "transform_time": 1.88651704788208
            infostr = m.group(1).strip()
            if not infostr.endswith('}'): infostr += '}'
            tmpdata = eval(infostr.replace('true','1').replace('false','0').replace('Infinity','float("inf")'))
            cost = tmpdata['cost']
            mintm, maxtm, avgtm = min(cost), max(cost), sum(cost)/len(cost)
            # Here we can choose to consider min, max, avg or something else
            dtime = avgtm
            paramvalues = tmpdata['perf_params']
            for k,v in paramvalues.items(): 
                if type(v) == bool: 
                    if v: paramvalues[k] = 1 
                    else: paramvalues[k] = 0
            datalist.append((paramvalues,(mintm,maxtm,avgtm)))
            if dtime < mintime: mintime = dtime
            if dtime > maxtime: maxtime = dtime
            
    
    for d in datalist:
        params=d[0]
        mintm,maxtm,avgtm = d[1]      # (mintime, maxtime, avgtime) triplet
        dtime = avgtm
        if dtime != float("inf") and dtime <= (1.0+besttol) * mintime: label = 'best'
        elif dtime != float("inf") and dtime >= (1.0+fairtol) * mintime: label = 'fair'
        else: label = 'worst'
        for v in params.values():
            buf += str(v) + ','
        if includetimes:
            for v in d[1]: buf += str(v) + ','
        buf += label + '\n'
        
        buf = buf.replace('inf',str(sys.float_info.max))
        
    return buf

def writeToFile(buf, fname):
    with open(os.path.basename(fname) + '2.arff', "w") as arff_file:
        arff_file.write(buf)
        arff_file.close()
    return 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', 
                        help="The file name of the Orio tuning log", type=str)
    parser.add_argument('-b', '--besttol', default=20,
                        help='The tolerance for assigning "best" to a time, e.g.,'+\
                              'if 20 is specified, then all times within 20%% of the minimum'+\
                              'will be assigned the "best" label',
                        type=int)
    parser.add_argument('-r', '--fairtol', default=40,
                        help='The tolerance for assigning "fair" to a time, e.g.,'+\
                              'if 40 is specified, then all times greater than the "best"'+\
                              'criterion but within 40%% of the minimum'+\
                              'will be assigned the "fair" label',
                        type=int)
    parser.add_argument('-t', '--times', default=False, 
                        help='If True, include the minimum, maximum, and average times in the'+\
                              'features.', action='store_true')

    args = parser.parse_args()

    fname = args.file
    besttol = args.besttol / 100.0
    fairtol = args.fairtol / 100.0
    includetimes = args.times   

    if not fname or not os.path.exists(fname): 
        print "Error: Please specify a valid Orio log file name."
        parser.print_usage(sys.stderr)
        sys.exit(1)
    
    lines = readTuningLog(fname)
    buf = convertToARFF(lines,besttol,fairtol,includetimes)
    writeToFile(buf,fname)
    pass