'''
Created on May 24, 2012

@author: mac
'''
from orio.main.util.globals import *
import time   
    
def recCoords(x, y, info, progress):
    ''' record x and y coordinates in a format that could be read in by Matlab/octave
    progress
    init: first record
    done: last record
    continue: any record in between init and done
    
    
    return elapsed time in writing record to file
    '''
    
    start_time = time.time()
    
    xfile = open('xfile.m', 'a')
    yfile = open('yfile.m', 'a')
    info = open('info.m', 'a')
    
    if progress == 'init':
        xfile.write('x = [%s' % (x))
        yfile.write('y = [%s' % (y))
        info.write('info = {%s' % (info))
        
    elif progress == 'continue':
        xfile.write(', %s' % (x))
        yfile.write(', %s' % (y))
        info.write(', %s' % (info))
        
    elif progress == 'done':
        xfile.write(', %s];' % (x))
        yfile.write(', %s];' % (y))
        info.write(', %s};' % (info))
        xfile.close()
        yfile.close()
        info.close()
    else:
        err('stat_record.recCoords: bad progress!!')
        
        
    IOtime = time.time()-start_time
    
    return IOtime
        
        