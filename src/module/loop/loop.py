#
# The class for loop transformation module
#

import sys
from orio.main.util.globals import *

from orio.module.module import Module

import codegen, parser, transformation

#-----------------------------------------

class Loop(Module):
    '''Loop transformation module'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='C'):
        '''Instantiate a loop transformation module'''
        
        Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)

    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply loop transformations on the annotated code'''

        # parse the code to get the AST
        stmts = parser.getParser(self.line_no).parse(self.module_body_code)

        # apply transformations
        t = transformation.Transformation(self.perf_params, self.verbose, self.language)
        transformed_stmts = t.transform(stmts)
        
        # generate code for the transformed ASTs
        indent = ' ' * self.indent_size
        extra_indent = '  '
        cgen = codegen.CodeGen(self.language)
        transformed_code = '\n'
        for s in transformed_stmts:
            transformed_code += cgen.generate(s, indent, extra_indent)

        # return the transformed code
        return transformed_code

