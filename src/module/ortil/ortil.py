#
# The main file (and class) for the tiling transformation module
#

import sys
import ann_parser, code_parser, module.module, pprinter, semant, transformator

#-----------------------------------------

class OrTil(module.module.Module):
    '''The class definition for the tiling transformation module'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code, cmd_line_opts,
                 line_no, indent_size):
        '''To instantiate a tiling transformation module'''
        
        module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      cmd_line_opts, line_no, indent_size)
        
    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply loop tiling on the annotated code'''

        # parse the text in the annotation module body to extract tiling information
        tiling_info = ann_parser.AnnParser(self.perf_params).parse(self.module_body_code)

        # parse the code (in the annotation body) to extract the corresponding AST
        stmts = code_parser.getParser().parse(self.annot_body_code)

        # analyze the AST semantics
        stmts = semant.SemanticAnalyzer(tiling_info).analyze(stmts)

        # perform loop-tiling transformation
        t = transformator.Transformator(tiling_info)
        (stmts, int_vars) = t.transform(stmts)

        # generate the declaration code for the newly declared integer variables
        code = ''
        for i,iv in enumerate(int_vars):
            if i%8 == 0:
                code += '\n  int '
            code += iv
            if i%8 == 7 or i == len(int_vars)-1:
                code += ';'
            else:
                code += ','
        if int_vars:
            code += '\n\n'

        # generate the tiled code
        for s in stmts:
            code += pprinter.PrettyPrinter().pprint(s)

        # append and prepend newlines (if necessary)
        if code[0] != '\n':
            code = '\n' + code
        if code[-1] != '\n':
            code = code + '\n'

        # return the tiled code
        return code

