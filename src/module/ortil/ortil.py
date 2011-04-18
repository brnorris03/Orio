#
# The orio.main.file (and class) for the tiling transformation module
# ORTIL = ORio TILing
#

import sys
import ann_parser, code_parser, orio.module.module, pprinter, semant, transformation

#-----------------------------------------

class OrTil(orio.module.module.Module):
    '''The class definition for OrTil tiling module.'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='C'):
        '''To instantiate an OrTil tiling module.'''
        
        orio.module.module.Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)
        
    #---------------------------------------------------------------------
    
    def __generateCode(self, stmts, int_vars):
        '''To generate the tiled loop code'''

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

        # return the generated code
        return code

    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply loop tiling on the annotated code'''

        # parse the text in the annotation orio.module.body to extract tiling information
        tiling_info = ann_parser.AnnParser(self.perf_params).parse(self.module_body_code)

        # parse the code (in the annotation body) to extract the corresponding AST
        stmts = code_parser.getParser().parse(self.annot_body_code)

        # analyze the AST semantics
        stmts = semant.SemanticAnalyzer(tiling_info).analyze(stmts)

        # perform loop-tiling transformation
        t = transformation.Transformation(tiling_info)
        (stmts, int_vars) = t.transform(stmts)

        # generate the tiled code
        code = self.__generateCode(stmts, int_vars)

        # return the tiled code
        return code

