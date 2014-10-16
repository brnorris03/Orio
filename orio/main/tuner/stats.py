'''
Rewritten from original version by mac

@author: norris

'''
from orio.main.util.globals import *
import time   

class Stats:
    def __init__(self,params={}):
        pass

    def record(self, x, y, extra="", progress=""):
        ''' record x and y coordinates -- subclasses implement different formats
        record extra information that corresponds to (x,y) as well. By default, there's no extra info (that is, empty string)
        progress
        init: first record
        done: last record
        continue: any record in between init and done
        
        return elapsed time in writing record to file
        '''
        raise NotImplementedError('%s: unimplemented abstract function "record"' %
                                  self.__class__.__name__)
    
class MatlabStats(Stats):
    def __init__(self, params={}):
        Stats.__init__(self,params)

    # Old name recCoords (not in a class)
    def record(self, x, y, extra="", progress=""):
        ''' record x and y coordinates in a format that could be read in by Matlab/octave
        record extra information that corresponds to (x,y) as well. By default, there's no extra info (that is, empty string)
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
            info.write('info = {"%s"' % (extra))
            
        elif progress == 'continue':
            xfile.write(', %s' % (x))
            yfile.write(', %s' % (y))
            info.write(', "%s"' % (extra))
            
        elif progress == 'done':
            xfile.write(', %s];' % (x))
            yfile.write(', %s];' % (y))
            info.write(', "%s"};' % (extra))
            xfile.close()
            yfile.close()
            info.close()
        else:
            err('Search incomplete, no stats recorded',self,doexit=False)
            
        IOtime = time.time()-start_time
        
        return IOtime
            
        
