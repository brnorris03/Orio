#
# The class for the Blue Gene's memory-alignment transformation module
#

import sys
import codegen, orio.module.module, parser

#-----------------------------------------

class Align(orio.module.module.Module):
    '''Memory-alignment transformation module.'''

    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='c'):
        '''To instantiate a memory-alignment transformation module.'''

        orio.module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)

    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply a memory-alignment transformation on the annotated code'''
 
        # parse the annotation orio.module.code to get the variables to be checked
        vars = parser.getParser(self.line_no).parse(self.module_body_code)

        # perform a semantic check
        for v in vars:
            v.semantCheck()

        # generate the alignment optimization code
        indent = ' ' * self.indent_size
        transformed_code = codegen.CodeGen(vars, self.annot_body_code, indent, self.language).generate()

        # return the transformed code
        return transformed_code

        
