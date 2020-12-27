import orio.module.loops.ast as ast

#----------------------------------------------------------------------------------------------------------------------
class Transformation:
    '''Insertion of pragma directives.'''

    def __init__(self, pragmas, stmt):

        self.pragmas = pragmas
        self.stmt = stmt

        # remove all empty pragma strings
        self.pragmas = [x for x in self.pragmas if x.strip()]


    def transform(self):

        # no pragma directives insertions
        if len(self.pragmas) == 0:
            return self.stmt

        # create a pragma directive AST
        prags = [ast.Pragma(p) for p in self.pragmas]

        # create the transformed statement
        if isinstance(self.stmt, ast.CompStmt):
            stmts = self.stmt.kids
        else:
            stmts = [self.stmt]
        transformed_stmt = ast.CompStmt(prags + stmts)

        # return the transformed statement
        return transformed_stmt
#----------------------------------------------------------------------------------------------------------------------


