#
# The class for loop transformation module
#

import sys, traceback
from orio.main.util.globals import *

from orio.module.module import Module

from orio.module.loop import codegen, parser, transformation, ast, astvisitors

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
        debug("Begin module.loop.loop.Loop.transform()", obj=self)
        theparser = parser.getParser(self.line_no)
        stmts = theparser.parse(self.module_body_code)
        debug("Successfully parsed the code starting at line %d:\n%s" % (self.line_no,str(stmts)), obj=self)
        if isinstance(stmts[0], ast.TransformStmt) and stmts[0].stmt is None:
            # transform the enclosed annot_body_code
            annotated_stmts = parser.getParser(self.line_no).parse(self.annot_body_code)
            if len(annotated_stmts) == 1:
                annotated_stmt = annotated_stmts[0]
            else:
                annotated_stmt = ast.CompStmt(annotated_stmts[0])
            stmts[0].stmt = annotated_stmt

        # apply transformations
        debug("Before Transformation constructor", obj=self)
        t = transformation.Transformation(self.perf_params, self.verbose, self.language, self.tinfo)
        debug("About to transform the code", obj=self)
        transformed_stmts = t.transform(stmts)
        
        
        # generate code for the transformed ASTs
        indent = ' ' * self.indent_size
        extra_indent = '  '
        cgen = codegen.CodeGen(self.language)
        transformed_code = '\n'
        for s in transformed_stmts:
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
            except Exception as e:
                err('[module.loop.loop] cannot construct CFG: ',e)


        # return the transformed code
        return transformed_code

