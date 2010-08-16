#
# The transformation that applies code transformation procedures
#

import sys
import ast, main.dyn_loader, orio.module.loop.codegen
from orio.main.util.globals import *

#-----------------------------------------

# the name of the transformation submodule
TSUBMOD_NAME = 'orio.module.loop.submodule'

#-----------------------------------------

class Transformation:
    '''Code transformation implementation'''

    def __init__(self, perf_params, verbose, language='C'):
        '''To instantiate a code transformation object'''
        self.perf_params = perf_params
        self.verbose = verbose
        self.language = language
        self.dloader = main.dyn_loader.DynLoader()
        
    #--------------------------------------

    def transform(self, stmts):
        '''Apply code transformations on each statement in the given statement list'''

        return [self.__transformStmt(s) for s in stmts]

    #--------------------------------------
    # Private methods
    #--------------------------------------
    

    def __transformStmt(self, stmt):
        '''Apply code transformation on the given statement'''
 
        if isinstance(stmt, ast.ExpStmt):
            return stmt

        elif isinstance(stmt, ast.CompStmt):
            stmt.stmts = [self.__transformStmt(s) for s in stmt.stmts]
            return stmt

        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__transformStmt(stmt.true_stmt)
            if stmt.false_stmt:
                stmt.false_stmt = self.__transformStmt(stmt.false_stmt)
            return stmt

        elif isinstance(stmt, ast.ForStmt):
            stmt.stmt = self.__transformStmt(stmt.stmt)
            return stmt

        elif isinstance(stmt, ast.TransformStmt):

            # transform the nested statement
            stmt.stmt = self.__transformStmt(stmt.stmt)

            # check for repeated transformation argument names
            arg_names = {}
            for [aname, rhs, line_no] in stmt.args:
                if aname in arg_names:
                    err('module.loop.transformation: %s: repeated transformation argument: "%s"' % (line_no, aname))
                arg_names[aname] = None

            # dynamically load the transformation submodule class
            class_name = stmt.name
            submod_name = '.'.join([TSUBMOD_NAME, class_name.lower(), class_name.lower()])
            submod_class = self.dloader.loadClass(submod_name, class_name)
            
            # apply code transformations
            try:
                t = submod_class(self.perf_params, stmt.args, stmt.stmt, self.language)
                transformed_stmt = t.transform()
            except Exception, e:
                err(('module.loop.transformation:%s: encountered an error as optimizing the transformation ' +
                        'statement: "%s"\n --> %s: %s') % (stmt.line_no, class_name,e.__class__.__name__, e))

            # return the transformed statement
            return transformed_stmt

        else:
            err('module.loop.transformation internal error: unknown statement type: %s' % stmt.__class__.__name__)
   
        

