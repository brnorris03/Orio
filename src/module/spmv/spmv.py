#
# The class for the sparse matrix-vector multiplication transformation module
#

import sys
import arg_info, module.module, parser, codegen

#-----------------------------------------

class SpMV(module.module.Module):
    '''SpMV transformation module'''

    def __init__(self, perf_params, module_body_code, annot_body_code, cmd_line_opts,
                 line_no, indent_size):
        '''To instantiate an SpMV transformation module'''
        
        module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      cmd_line_opts, line_no, indent_size)

    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply an SpMV transformation on the annotated code'''

        # parse the module body code
        args = parser.Parser().parse(self.module_body_code, self.line_no)
        
        # generate the input argument information
        ainfo = arg_info.ArgInfoGen().generate(args, self.perf_params)

        # generate the optimized code
        optimized_code = codegen.CodeGen(ainfo).generate()

        # return the optimized code
        return optimized_code

        
