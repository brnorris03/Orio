#
# File: src/orio.module.module.py
#

from orio.main.util.globals import *

class Module:
    '''The abstract class of Orio's code transformation module.'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='c'):
        '''
        The class constructor used to instantiate a program transformation orio.module.
        
        The following are the class attributes:
          perf_params         a table that maps each performance parameter to its value
          module_body_code    the code inside the orio.module.body block
          annot_body_code     the code contained in the annotation body block
          line_no             the starting line position of the orio.module.code in the source code
          indent_size         an integer representing the number of whitespace characters that
                              preceed the leader annotation
          language            the language of the input code (C or Fortran)
        '''

        self.perf_params = perf_params
        self.module_body_code = module_body_code
        self.annot_body_code = annot_body_code
        self.line_no = line_no
        self.language = language
        self.indent_size = indent_size
        
        # a boolean value to indicate if the results of the running transformation need to be shown
        self.verbose = Globals().verbose

    #--------------------------------------------------------

    def transform(self):
        '''
        The orio.main.code transformation procedure. The returned value is a string value that
        represents the transformed/optimized code.
        '''

        raise NotImplementedError('%s: unimplemented abstract function "transform"' %
                                  (self.__class__.__name__)) 

