from orio.module.module import Module
import orio.module.loops.printer as printer
import orio.module.loops.parser as parser
import orio.module.loops.transformation as rewriter
import orio.module.loops.ast as ast

#----------------------------------------------------------------------------------------------------------------------
class Loops(Module):
    '''Loops transformation module'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='C', tinfo=None):
        
        Module.__init__(self, perf_params, module_body_code, annot_body_code,
                        line_no, indent_size, language)
        self.tinfo = tinfo


    def transform(self):
        '''Applies loop transformations on the annotated code'''

        # parse the code to get the AST
        stmts = parser.parse(self.line_no, self.module_body_code)
        if isinstance(stmts[0], ast.TransformStmt) and stmts[0].stmt is None:
            # transform the enclosed annot_body_code
            annotated_stmts = parser.parse(self.line_no, self.annot_body_code)
            if len(annotated_stmts) == 1:
                annotated_stmt = annotated_stmts[0]
            else:
                annotated_stmt = ast.CompStmt(annotated_stmts[0])
            stmts[0].stmt = annotated_stmt

        # apply transformations
        t = rewriter.Transformation(self.perf_params, self.verbose, self.language, self.tinfo)
        transformed_stmts = t.transform(stmts)
        
        # generate code for the transformed ASTs
        cgen = printer.CodeGen(self.language)
        indent = ' ' * self.indent_size
        extra_indent = '  '
        transformed_code = '\n'
        for s in transformed_stmts:
            transformed_code += cgen.generate(s, indent, extra_indent)

        # return the transformed code
        return transformed_code
#----------------------------------------------------------------------------------------------------------------------


