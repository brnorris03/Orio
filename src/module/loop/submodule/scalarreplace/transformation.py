#
# Contain the transformation procedure
#

import sys
from orio.main.util.globals import *
import orio.module.loop.ast, orio.module.loop.ast_lib.common_lib, orio.module.loop.ast_lib.forloop_lib

#-----------------------------------------

class Transformation:
    '''Code transformation implementation'''

    def __init__(self, dtype, prefix, stmt):
        '''Instantiate a code transformation object'''

        self.dtype = 'double'
        self.prefix = 'scv_'
        if dtype != None:
            self.dtype = dtype
        if prefix != None:
            self.prefix = prefix
        self.stmt = stmt

        self.counter = 1
        self.flib = orio.module.loop.ast_lib.forloop_lib.ForLoopLib()
        self.clib = orio.module.loop.ast_lib.common_lib.CommonLib()
        
    #----------------------------------------------------------

    def __collectRefs(self, exp, refs_map, is_output = False):
        '''Return the array references from the given expression'''

        if exp == None:
            return
        
        if isinstance(exp, orio.module.loop.ast.NumLitExp):
            return
        
        elif isinstance(exp, orio.module.loop.ast.StringLitExp):
            return
        
        elif isinstance(exp, orio.module.loop.ast.IdentExp):
            return

        elif isinstance(exp, orio.module.loop.ast.ArrayRefExp):
            rkey = str(exp)
            if rkey in refs_map:
                ifreq, iisoutput, iexp = refs_map[rkey]
                refs_map[rkey] = (ifreq+1, iisoutput or is_output, iexp)
            else:
                refs_map[rkey] = (1, is_output, exp.replicate())

        elif isinstance(exp, orio.module.loop.ast.FunCallExp):
            self.__collectRefs(exp.exp, refs_map, is_output)
            for a in exp.args:
                self.__collectRefs(a, refs_map, is_output)
            
        elif isinstance(exp, orio.module.loop.ast.UnaryExp):
            self.__collectRefs(exp.exp, refs_map, is_output)
            
        elif isinstance(exp, orio.module.loop.ast.BinOpExp):
            if exp.op_type == orio.module.loop.ast.BinOpExp.EQ_ASGN:
                self.__collectRefs(exp.lhs, refs_map, True)
            else:
                self.__collectRefs(exp.lhs, refs_map, is_output)
            self.__collectRefs(exp.rhs, refs_map, is_output)

        elif isinstance(exp, orio.module.loop.ast.ParenthExp):
            self.__collectRefs(exp.exp, refs_map, is_output)

        elif isinstance(exp, orio.module.loop.ast.NewAST):
            return
        
        elif isinstance(exp, orio.module.loop.ast.Comment):
            return  
                
        else:
            err('orio.module.loop.submodule.scalarreplace.transformation internal error: unexpected AST type: "%s"' % exp.__class__.__name__)
            
    #----------------------------------------------------------

    def __createScalars(self, scalars):
        '''Create prologue and epilogue to declare, initialize, and update the scalars'''

        # create declarations, initializations, and updates
        # (also the mapping from references to their corresponding scalars)
        decls = []
        inits = []
        updates = []
        scalars_map = {}
        last_decl = None
        for isoutput, exp in scalars:

            # create a new variable name
            vname = self.prefix + str(self.counter)

            # create declaration
            if last_decl == None:
                last_decl = orio.module.loop.ast.VarDecl(self.dtype, [vname])
            elif len(last_decl.var_names) >= 8:
                decls.append(last_decl)
                last_decl = orio.module.loop.ast.VarDecl(self.dtype, [vname])
            else:
                last_decl.var_names.append(vname)

            # create initialization
            iexp = orio.module.loop.ast.BinOpExp(orio.module.loop.ast.IdentExp(vname),
                                            exp.replicate(),
                                            orio.module.loop.ast.BinOpExp.EQ_ASGN)
            inits.append(orio.module.loop.ast.ExpStmt(iexp))
            
            # create updates, if needed
            if isoutput:
                iexp = orio.module.loop.ast.BinOpExp(exp.replicate(),
                                                orio.module.loop.ast.IdentExp(vname),
                                                orio.module.loop.ast.BinOpExp.EQ_ASGN)
                updates.append(orio.module.loop.ast.ExpStmt(iexp))

            # update the scalars mapping
            scalars_map[str(exp)] = orio.module.loop.ast.IdentExp(vname)
                
            # increment the counter
            self.counter += 1

        # append the last declaration
        if last_decl:
            decls.append(last_decl)

        # create prologue and epilogue 
        prologue = decls + inits
        epilogue = updates

        # return prologue and epilogue, and the scalars mapping
        return (prologue, epilogue, scalars_map)

    #----------------------------------------------------------

    def __replaceRefs(self, tnode, scalars_map):
        '''
        To replace all matching array references with the specified scalars
        '''

        if tnode == None:
            return None

        if isinstance(tnode, orio.module.loop.ast.NumLitExp):
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.StringLitExp):
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.IdentExp):
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.ArrayRefExp):
            rkey = str(tnode)
            if rkey in scalars_map:
                return scalars_map[rkey].replicate()
            else:
                return tnode

        elif isinstance(tnode, orio.module.loop.ast.FunCallExp):
            tnode.exp = self.__replaceRefs(tnode.exp, scalars_map)
            tnode.args = [self.__replaceRefs(a, scalars_map) for a in tnode.args]
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.UnaryExp):
            tnode.exp = self.__replaceRefs(tnode.exp, scalars_map)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.BinOpExp):
            tnode.lhs = self.__replaceRefs(tnode.lhs, scalars_map)
            tnode.rhs = self.__replaceRefs(tnode.rhs, scalars_map)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.ParenthExp):
            tnode.exp = self.__replaceRefs(tnode.exp, scalars_map)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.ExpStmt):
            tnode.exp = self.__replaceRefs(tnode.exp, scalars_map)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.CompStmt):
            tnode.stmts = [self.__replaceRefs(s, scalars_map) for s in tnode.stmts]
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.IfStmt):
            tnode.test = self.__replaceRefs(tnode.test, scalars_map)
            tnode.true_stmt = self.__replaceRefs(tnode.true_stmt, scalars_map)
            tnode.false_stmt = self.__replaceRefs(tnode.false_stmt, scalars_map)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.ForStmt):
            tnode.stmt = self.__replaceRefs(tnode.stmt, scalars_map)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.scalarreplace.transformation internal error: unprocessed transform statement')

        elif isinstance(tnode, orio.module.loop.ast.NewAST):
            return tnode
        
        elif isinstance(tnode, orio.module.loop.ast.Comment):
            return tnode 
        
        else:
            err('orio.module.loop.submodule.scalarreplace.transformation internal error: unexpected AST type: "%s"' % tnode.__class__.__name__)

    #----------------------------------------------------------

    def __replaceScalars(self, stmt, refs_map):
        '''Replace array references in the given statement with scalars'''

        if stmt == None:
            return

        if isinstance(stmt, orio.module.loop.ast.ExpStmt):
            self.__collectRefs(stmt.exp, refs_map)

        elif isinstance(stmt, orio.module.loop.ast.CompStmt):
            for s in stmt.stmts:
                self.__replaceScalars(s, refs_map)
                
        elif isinstance(stmt, orio.module.loop.ast.IfStmt):
            self.__collectRefs(stmt.test, refs_map)
            self.__replaceScalars(stmt.true_stmt, refs_map)
            self.__replaceScalars(stmt.false_stmt, refs_map)
            
        elif isinstance(stmt, orio.module.loop.ast.ForStmt):

            # collect array references from the loop body
            self.__replaceScalars(stmt.stmt, refs_map)

            # get the current loop index name
            for_loop_info = self.flib.extractForLoopInfo(stmt)
            index_id, lbound_exp, ubound_exp, stride_exp, loop_body = for_loop_info
            iname = index_id.name

            # filter out references that contain the current loop's index name, to form scalars
            scalars = []
            irefs_map = {}
            for rkey, (freq, isoutput, exp) in refs_map.iteritems():
                if self.clib.containIdentName(exp, iname):
                    if freq > 1:
                        scalars.append((isoutput, exp))
                else:
                    irefs_map[rkey] = (freq, isoutput, exp)
            refs_map.clear()
            refs_map.update(irefs_map.items())

            # create prologue and epilogue (to declare, initialize, and update the scalars), and
            # the scalars mapping
            prologue, epilogue, scalars_map = self.__createScalars(scalars)

            # replace all references with the formed scalars
            stmt.stmt = self.__replaceRefs(stmt.stmt, scalars_map)

            # insert prologue and epilogue to the loop body
            if isinstance(stmt.stmt, orio.module.loop.ast.CompStmt):
                stmt.stmt.stmts = prologue + stmt.stmt.stmts + epilogue
            else:
                stmt.stmt = orio.module.loop.ast.CompStmt(prologue + [stmt.stmt] + epilogue)
            
        elif isinstance(stmt, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.submodule.scalarreplace.transformation internal error: unprocessed transform statement')
            
        elif isinstance(stmt, orio.module.loop.ast.NewAST):
            return

        elif isinstance(stmt, orio.module.loop.ast.Comment):
            return
               
        else:
            err('orio.module.loop.submodule.scalarreplace.transformation internal error: unexpected AST type: "%s"' % stmt.__class__.__name__)
            
    #----------------------------------------------------------

    def transform(self):
        '''To replace array references with scalars'''
        
        # reset counter
        self.counter = 1

        # copy the statement to be transformed
        transformed_stmt = self.stmt.replicate()
        
        # perform scalar replacement
        refs_map = {}
        self.__replaceScalars(transformed_stmt, refs_map)

        # find all the reorio.main.ng scalars
        scalars = []
        for rkey, (freq, isoutput, exp) in refs_map.iteritems():
            if freq > 1:
                scalars.append((isoutput, exp))

        # create prologue and epilogue (to declare, initialize, and update the scalars), and
        # the scalars mapping
        prologue, epilogue, scalars_map = self.__createScalars(scalars)

        # replace all references with the formed scalars
        transformed_stmt = self.__replaceRefs(transformed_stmt, scalars_map)

        # insert prologue and epilogue to the loop body
        if isinstance(transformed_stmt, orio.module.loop.ast.CompStmt):
            transformed_stmt.stmts = prologue + transformed_stmt.stmts + epilogue
        else:
            transformed_stmt = orio.module.loop.ast.CompStmt(prologue + [transformed_stmt] + epilogue)

        # return the transformed statement
        return transformed_stmt
