'''
Rewritten from original version by mac

@author: norris

'''
from orio.main.util.globals import *
import time   

class Stats:
    def __init__(self,filename=None,params={}):
        '''
        @param filename: an optional filename for statistics output
        @param params: a dictionary of name-value pairs for configuring different Stats subclasses
        '''
        self.filename = filename
        if self.filename: 
            fileDesc = open(self.filename,'w')
            fileDesc.close()
        self.buf = ''
        pass
    
    def write(self, info, force=False):
        self.buf += self.record(info)
        # Buffer output to reduce file I/O overhead
        if len(self.buf) < 1024 and not force: return
        if self.filename: fileDesc = open(self.filename,'a')
        else: fileDesc = sys.stdout
        fileDesc.write(self.record(info))      # implemented by subclasses
        if fileDesc is not sys.stdout: fileDesc.close()
        self.buf = ''
        pass

    def record(self, coord, metrics, counter=None, extra=""):
        ''' Record each coordinate and corresponding measurement, e.g., time.
        @param coord: the list of current parameter values
        @param metrics: a dictionary of name-value pairs containing the names and values of
             measurements corresponding to this coordinate
        @param counter: an integer counter (for indexing variants)
        @param extra: an optional extra string to include in output 
        @return: a string 
        '''
        raise NotImplementedError('%s: unimplemented abstract function "record"' %
                                  self.__class__.__name__)

class Simple(Stats):
    def __init__(self, filename=None, params={}):
        Stats.__init__(self,params)
        self.sep = ';'   # separator for separating different entities being output
        self.first = True
        pass
        
    def record(self, coord, metrics, counter = None, extra={}):
        buf = ''
        if self.first: 
            if counter: buf = 'Id' + self.sep
            buf += self.sep.join(['Coordinate']+metrics.keys()) + self.sep.join(extra.keys()) + '\n'
            self.first = False
        if counter: buf += str(counter) + self.sep
        buf += self.sep.join([str(coord)]+metrics.values()) 
        buf += self.sep.join(extra.values()) 
        buf += '\n'
        return buf
    
# -----------------------------------------------------------------------------

class Custom(Stats):
    '''
    Read a config file to create a custom stats output format
    '''
    def __init__(self, filaname=None):
        pass
    
    def record(self, coord, metrics, counter = None, extra = {}):
        pass
    
# -----------------------------------------------------------------------------

class Matlab(Stats):
    def __init__(self, params={}):
        Stats.__init__(self,params)

    # Old name recCoords (not in a class)
    def record(self, self, coord, metrics, counter=None, extra=""):
        
        start_time = time.time()
        # TODO: rewrite to return a string
        
#         xfile = open('xfile.m', 'a')
#         yfile = open('yfile.m', 'a')
#         info = open('info.m', 'a')
#         
#         if progress == 'init':
#             xfile.write('x = [%s' % (x))
#             yfile.write('y = [%s' % (y))
#             info.write('info = {"%s"' % (extra))
#             
#         elif progress == 'continue':
#             xfile.write(', %s' % (x))
#             yfile.write(', %s' % (y))
#             info.write(', "%s"' % (extra))
#             
#         elif progress == 'done':
#             xfile.write(', %s];' % (x))
#             yfile.write(', %s];' % (y))
#             info.write(', "%s"};' % (extra))
#             xfile.close()
#             yfile.close()
#             info.close()
#         else:
#             err('Search incomplete, no stats recorded',self,doexit=False)
#             
        IOtime = time.time()-start_time
        
        return IOtime
            
        
