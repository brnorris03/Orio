#
# The class for loop transformation module
#

import sys, traceback
from orio.main.util.globals import *

from orio.module.module import Module

import codegen, parser, transformation, ast
from orio.module.loop import astvisitors

#-----------------------------------------

class Loop(Module):
    '''Loop transformation module'''
    
    def __init__(self, perf_params, module_body_code, annot_body_code,
                 line_no, indent_size, language='C', tinfo=None):
        '''Instantiate a loop transformation module'''
        
        Module.__init__(self, perf_params, module_body_code, annot_body_code,
                                      line_no, indent_size, language)
        self.tinfo = tinfo

    #---------------------------------------------------------------------
    
    def transform(self):
        '''To apply loop transformations on the annotated code'''

        # parse the code to get the AST
        stmts = parser.getParser(self.line_no).parse(self.module_body_code)
        if isinstance(stmts[0], ast.TransformStmt) and stmts[0].stmt is None:
            # transform the enclosed annot_body_code
            annotated_stmts = parser.getParser(self.line_no).parse(self.annot_body_code)
            if len(annotated_stmts) == 1:
                annotated_stmt = annotated_stmts[0]
            else:
                annotated_stmt = ast.CompStmt(annotated_stmts[0])
            stmts[0].stmt = annotated_stmt

        # apply transformations
        t = transformation.Transformation(self.perf_params, self.verbose, self.language, self.tinfo)
        transformed_stmts = t.transform(stmts)
        
        debug("orio.module.loop.transform: after transformation, before code gen", obj=self)
        # generate code for the transformed ASTs
        indent = ' ' * self.indent_size
        extra_indent = '  '
        cgen = codegen.CodeGen(self.language)
        transformed_code = '\n'
        for s in transformed_stmts:
            debug("orio.module.loop.transform: before generating code for %s" % str(s.__class__),obj=self)
            transformed_code += cgen.generate(s, indent, extra_indent)
            
        # Example on applying another visitor, e.g., for analysis
        #exampleVisitor = astvisitors.ExampleVisitor()
        #exampleVisitor.visit(transformed_stmts)
        
        # Count operations visitor
        opsVisitor = astvisitors.CountingVisitor()
        opsVisitor.visit(transformed_stmts)
        debug(str(opsVisitor),level=3)
        
        # CFG
        if True:
            try:
                from orio.module.loop.cfg import CFGGraph
                cfg = CFGGraph(transformed_stmts)
            except Exception, e:
                err('[module.loop.loop] cannot construct CFG: ',e)


        # return the transformed code
        return transformed_code

