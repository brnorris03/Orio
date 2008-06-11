#
# The abstract class of the transformation module
#

class Module:
    '''Transformation module'''
    
    def __init__(self, cmd_line_opts, perf_params, module_code, line_no,
                 indent_size, annot_body_code):
        '''
        To instantiate a transformation module used to transform the annotated code.

        The class variables consist of the following:
           cmd_line_opts      an object representing the command line options
                              (see src/cmd_line_opts.py for more details)
           perf_params        a table/mapping that maps each performance parameter to its value
           module_code        the code of annotation module body
           line_no            the starting line position of the module code in the source code
           indent_size        an integer representing the number of space characters used in the
                              indentation of the leader annotation
           annot_body_code    the transformed/optimized code of the annotation body
           verbose            to show details of the results of the running transformation modules
        '''

        self.cmd_line_opts = cmd_line_opts
        self.perf_params = perf_params
        self.module_code = module_code
        self.line_no = line_no
        self.indent_size = indent_size
        self.annot_body_code = annot_body_code

        # other derived class variables
        self.verbose = self.cmd_line_opts.verbose

    #--------------------------------------------------------------------
        
    def transform(self):
        '''
        The transformation procedure used to transform the annotated code.
        The returned value is a string that represents the transformed/optimized code.
        '''

        raise NotImplementedError('%s: unimplemented abstract function "transform"' %
                                  (self.__class__.__name__))

