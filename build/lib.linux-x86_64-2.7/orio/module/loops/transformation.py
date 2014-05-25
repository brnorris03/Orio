import orio.main.util.globals as g
import orio.module.loops.ast as ast
import orio.main.dyn_loader

#----------------------------------------------------------------------------------------------------------------------
# prefix of transformation submodules
TSUBMOD_NAME = 'orio.module.loops.submodule'


#----------------------------------------------------------------------------------------------------------------------
class Transformation:
    '''Code transformation implementation'''

    def __init__(self, perf_params, verbose, language='C', tinfo=None):
        self.dloader = orio.main.dyn_loader.DynLoader()
        self.perf_params = perf_params
        self.verbose = verbose
        self.language = language
        self.tinfo = tinfo


    def transform(self, stmts):
        '''Apply code transformations on each statement in the given statement list'''

        return [self.__transformStmt(s) for s in stmts]


    def __transformStmt(self, stmt):
        '''Apply code transformation on the given statement'''
        
        if isinstance(stmt, ast.ExpStmt):
            return stmt
        
        elif isinstance(stmt, ast.CompStmt):
            stmt.kids = [self.__transformStmt(s) for s in stmt.kids]
            return stmt

        elif isinstance(stmt, ast.IfStmt):
            stmt.true_stmt = self.__transformStmt(stmt.true_stmt)
            if stmt.false_stmt:
                stmt.false_stmt = self.__transformStmt(stmt.false_stmt)
            return stmt

        elif isinstance(stmt, ast.ForStmt):
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
                    g.err(__name__ + ': %s: repeated transformation argument: "%s"' % (line_no, aname))
                arg_names[aname] = None

            # dynamically load the transformation submodule class
            class_name = stmt.name
            submod_name = '.'.join([TSUBMOD_NAME, class_name.lower(), class_name.lower()])
            submod_class = self.dloader.loadClass(submod_name, class_name)
            
            # apply code transformations
            t = submod_class(self.perf_params, stmt.args, stmt.stmt, self.language, self.tinfo)
            transformed_stmt = t.transform()

            return transformed_stmt

        else:
            g.err(__name__+': internal error: unknown statement type: %s' % stmt.__class__.__name__)
#----------------------------------------------------------------------------------------------------------------------
   
        

