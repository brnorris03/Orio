#
# The class for the sparse matrix-vector multiplication transformation module
#

import sys
import arg_info, orio.module.module, parser, codegen

#-----------------------------------------

class SpMV(orio.module.module.Module):
    '''SpMV transformation module.'''

    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='C'):
        '''To instantiate an SpMV transformation module.'''
        
        orio.module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)

    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply an SpMV transformation on the annotated code'''

        # parse the orio.module.body code
        args = parser.Parser().parse(self.module_body_code, self.line_no)
        
        # generate the input argument information
        ainfo = arg_info.ArgInfoGen().generate(args, self.perf_params)

        # generate the optimized code
        optimized_code = codegen.CodeGen(ainfo).generate()

        # return the optimized code
        return optimized_code

        
