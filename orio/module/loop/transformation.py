#
# The transformation that applies code transformation procedures
#

from orio.module.loop import ast
import orio.main.dyn_loader
from orio.main.util.globals import *

#-----------------------------------------

# the name of the transformation submodule
TSUBMOD_NAME = 'orio.module.loop.submodule'

#-----------------------------------------

class Transformation:
    '''Code transformation implementation'''

    def __init__(self, perf_params, verbose, language='C', tinfo=None):
        '''To instantiate a code transformation object'''
        self.perf_params = perf_params
        self.verbose = verbose
        self.language = language
        self.dloader = orio.main.dyn_loader.DynLoader()
        self.tinfo = tinfo
        
    #--------------------------------------

    def transform(self, stmts):
        '''Apply code transformations on each statement in the given statement list'''
        Globals().metadata['loop_transformations'] = []
        # for s in stmts:
        #     Globals().metadata["loop_transformations"].append(s.name)
        return [self.__transformStmt(s) for s in stmts]

    #--------------------------------------
    # Private methods
    #--------------------------------------
    

    def __transformStmt(self, stmt):
        '''Apply code transformation on the given statement'''

        try:
            if isinstance(stmt, ast.ExpStmt):
                return stmt
            
            if isinstance(stmt, ast.GotoStmt):
                return stmt
    
            elif isinstance(stmt, ast.CompStmt):
                if not stmt.meta.get('id'):
                    stmt.meta['id'] = 'loop_' + str(stmt.line_no)
                stmt.stmts = [self.__transformStmt(s) for s in stmt.stmts]
                return stmt
    
            elif isinstance(stmt, ast.IfStmt):
                stmt.true_stmt = self.__transformStmt(stmt.true_stmt)
                if stmt.false_stmt:
                    stmt.false_stmt = self.__transformStmt(stmt.false_stmt)
                return stmt
    
            elif isinstance(stmt, ast.ForStmt):
                if not stmt.parent or (stmt.parent and not stmt.parent.meta.get('id') and not stmt.meta.get('id')):
                    stmt.meta['id'] = 'loop_' + str(stmt.line_no)
                stmt.stmt = self.__transformStmt(stmt.stmt)
                return stmt
    
            elif isinstance(stmt, ast.Comment):
                stmt.stmt = stmt
                return stmt
            
            elif isinstance(stmt, ast.TransformStmt):
    
                # transform the nested statement
                stmt.stmt = self.__transformStmt(stmt.stmt)
    
                # check for repeated transformation argument names
                arg_names = {}
                for [aname, _, line_no] in stmt.args:
                    if aname in arg_names:
                        err('orio.module.loop.transformation: %s: repeated transformation argument: "%s"' % (line_no, aname))
                    arg_names[aname] = None
    
                # dynamically load the transformation submodule class
                class_name = stmt.name
                submod_name = '.'.join([TSUBMOD_NAME, class_name.lower(), class_name.lower()])
                submod_class = self.dloader.loadClass(submod_name, class_name)
                
                # apply code transformations
                try:
                    if self.language == 'cuda' or self.language == 'opencl':
                        t = submod_class(self.perf_params, stmt.args, stmt.stmt, self.language, self.tinfo)
                    else:
                        t = submod_class(self.perf_params, stmt.args, stmt.stmt, self.language)
                except Exception as e:
                    err('orio.module.loop.transformation could not load submodule %s: %s' % (submod_name,e))
                    
                try:
                    transformed_stmt = t.transform()
                except Exception as e:
                    err(('orio.module.loop.transformation:%s: encountered an error during transformation of ' +
                            'statement: "%s"\n --> %s: %s') % (stmt.line_no, class_name,e.__class__.__name__, e), 0, False)
                    raise Exception(e)
    
                # return the transformed statement
                return transformed_stmt
    
            else:
                err('orio.module.loop.transformation internal error: unknown statement type: %s' % stmt.__class__.__name__)
   
        except Exception as e:
            err('orio.module.loop.transformation exception for statement %s' % stmt.__class__.name)

