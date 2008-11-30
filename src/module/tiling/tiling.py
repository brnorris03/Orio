#
# The main file (and class) for the tiling transformation module
#

import sys
import ann_parser, code_parser, module.module, pprinter, semant, transformator

#-----------------------------------------

class Tiling(module.module.Module):
    '''The class definition for the tiling transformation module'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code, cmd_line_opts,
                 line_no, indent_size):
        '''To instantiate a tiling transformation module'''
        
        module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      cmd_line_opts, line_no, indent_size)
        
    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply loop tiling on the annotated code'''

        # parse the code (in the annotation body) to extract the corresponding AST
        code_stmts = code_parser.getParser().parse(self.annot_body_code)

        # analyze the AST semantics
        code_stmts = semant.SemanticAnalyzer().analyze(code_stmts)

        # parse the text in the annotation module body to extract tiling information
        tiling_info = ann_parser.AnnParser(self.perf_params).parse(self.module_body_code)

        # perform loop-tiling transformation
        code_stmts = transformator.Transformator(self.perf_params, tiling_info).transform(code_stmts)

        print '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-'
        print code_stmts
        sys.exit(1)
        
        # generate the code of the tiled code
        tiled_code = ''
        for s in code_stmts:
            tiled_code = tiled_code + pprinter.PrettyPrinter().pprint(s)
        if tiled_code[0] != '\n':
            tiled_code = '\n' + tiled_code
        if tiled_code[-1] != '\n':
            tiled_code = tiled_code + '\n'

        print tiled_code

        print '----- forced to exit -----'
        sys.exit(1)

        # return the tiled code
        return tiled_code

