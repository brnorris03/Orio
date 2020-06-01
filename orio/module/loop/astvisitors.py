'''
Created on Feb 26, 2015

@author: norris
'''
import ast, sys, os, traceback
from orio.main.util.globals import *

import orio.module.loop.codegen

class ASTVisitor:
    ''' Standard visitor pattern abstract class'''
    def __init__(self):
        self.verbose = False
        if 'LOOPAST_DEBUG' in os.environ.keys(): self.verbose = True
    
    def __str__(self):
        return self.__class__.__name__
    
    def display(self, node, msg=''):
        if self.verbose:
            sys.stdout.write("[%s] " % self.__class__.__name__ + node.__class__.__name__ + ': ' + msg+'\n')
    
    def visit(self, node, params={}):
        ''' Default visit method to be overwritten by specific visitors'''
        pass

class ExampleVisitor(ASTVisitor):
    def __init__(self):
        ASTVisitor.__init__(self)
        self.cgen = orio.module.loop.codegen.CodeGen_C()
        

    def visit(self, nodes, params={}):
        '''Invoke accept method for specified AST node'''
        if not isinstance(nodes, (list, tuple)):
            nodes = list(nodes)
            
        for node in nodes:
            try:
                if isinstance(node, ast.NumLitExp):
                    self.display(node, str(node.val))
        
                elif isinstance(node, ast.StringLitExp):
                    self.display(node, str(node.val))
        
                elif isinstance(node, ast.IdentExp):
                    self.display(node, str(node.name))
        
                elif isinstance(node, ast.ArrayRefExp):
                    self.display(node, self._generate(node.exp))
                    
                elif isinstance(node, ast.FunCallExp):
                    s = self._generate(node.exp) + '('
                    s += ','.join(map(lambda x: self._generate(x), node.args))
                    s += ')'
                    self.display(node, s)
        
                elif isinstance(node, ast.UnaryExp):
                    s = self._generate(node.exp)
                    if node.op_type == node.PLUS:
                        s = '+' + s
                    elif node.op_type == node.MINUS:
                        s = '-' + s
                    elif node.op_type == node.LNOT:
                        s = '!' + s
                    elif node.op_type == node.PRE_INC:
                        s = ' ++' + s
                    elif node.op_type == node.PRE_DEC:
                        s = ' --' + s
                    elif node.op_type == node.POST_INC:
                        s = s + '++ '
                    elif node.op_type == node.POST_DEC:
                        s = s + '-- '
                    elif node.op_type == node.DEREF:
                        s = '*' + s
                    elif node.op_type == node.ADDRESSOF:
                        s = '&' + s
                    else:
                        s = 'Unknown unary operator:'+ str(s.op_type)
                    self.display(node, s)
    
                elif isinstance(node, ast.BinOpExp):
                    s = self._generate(node.lhs)
                    if node.op_type == node.MUL:
                        s += '*'
                    elif node.op_type == node.DIV:
                        s += '/'
                    elif node.op_type == node.MOD:
                        s += '%'
                    elif node.op_type == node.ADD:
                        s += '+'
                    elif node.op_type == node.SUB:
                        s += '-'
                    elif node.op_type == node.LT:
                        s += '<'
                    elif node.op_type == node.GT:
                        s += '>'
                    elif node.op_type == node.LE:
                        s += '<='
                    elif node.op_type == node.GE:
                        s += '>='
                    elif node.op_type == node.EQ:
                        s += '=='
                    elif node.op_type == node.NE:
                        s += '!='
                    elif node.op_type == node.LOR:
                        s += '||'
                    elif node.op_type == node.LAND:
                        s += '&&'
                    elif node.op_type == node.COMMA:
                        s += ','
                    elif node.op_type == node.EQ_ASGN:
                        s += '='
                    else:
                        s = 'Unknown binary operator:'+ str(s.op_type)
                    self.display(node, s)
        
                elif isinstance(node, ast.ParenthExp):
                    s = '(' + self._generate(node.exp) + ')'
                    self.display(node, s)
        
                elif isinstance(node, ast.Comment):
                    s = '/*' + str(node.text) + '*/'
                    self.display(node, s)
                    
                elif isinstance(node, ast.ExpStmt):
                    s=''
                    if node.getLabel(): s += str(node.getLabel()) + ':'
                    if node.exp:
                        s += str(self._generate(node.exp))
                    self.display(node, s)
        
                elif isinstance(node, ast.GotoStmt):
                    s = ''
                    if node.getLabel(): s += node.getLabel() + ':'
                    if node.target:
                        s += 'goto ' + str(node.target) + ';\n'
                    self.display(node, s)
                        
                elif isinstance(node, ast.CompStmt):
                    self.alldecls = set([])
                    self.display(node)
                    self.visit(node.stmts)
    
                elif isinstance(node, ast.IfStmt):
                    s = ''
                    if node.getLabel(): s = str(node.getLabel()) + ':'
                    s += 'if (' + self._generate(node.test) + ') '
                    if isinstance(node.true_stmt, ast.CompStmt):
                        tstmt_s = self._generate(node.true_stmt)
                        s += tstmt_s[tstmt_s.index('{'):]
                        if node.false_stmt:
                            s = str(s[:-1]) + ' else '
                    else:
                        s += '\n'
                        s += self._generate(node.true_stmt)
                        if node.false_stmt:
                            s += 'else '
                    if node.false_stmt:
                        if isinstance(node.false_stmt, ast.CompStmt):
                            tstmt_s = self._generate(node.false_stmt)
                            s += tstmt_s[tstmt_s.index('{'):]
                        else:
                            s += '\n'
                            s += self._generate(node.false_stmt)
                    self.display(node, s)
                            
        
                elif isinstance(node, ast.ForStmt):
                    s=''
                    if node.getLabel(): s = str(node.getLabel()) + ':'
                    s += 'for ('
                    if node.init:
                        s += self._generate(node.init)
                    s += '; '
                    if node.test:
                        s += self._generate(node.test)
                    s += '; '
                    if node.iter:
                        s += self._generate(node.iter)
                    s += ') '
                    if isinstance(node.stmt, ast.CompStmt): 
                        stmt_s = self._generate(node.stmt)
                        s += stmt_s[stmt_s.index('{'):]
                        self.alldecls = set([])
                    else:
                        s += '\n'
                        s += self._generate(node.stmt)
                    self.display(node, s)
    
        
                elif isinstance(node, ast.TransformStmt):
                    self.display(node, "Internal error: a transformation statement is never generated as an output")
        
                elif isinstance(node, ast.VarDecl):
                    s =''
                    sv = str(node.type_name) + ' '
                    sv += ', '.join(node.var_names)
                    sv += ';\n'
                    if not sv in self.alldecls: 
                        s += sv
                        self.alldecls.add(sv)
                    self.display(node, s)
        
                elif isinstance(node, ast.VarDeclInit):
                    s = str(node.type_name) + ' '
                    s += self._generate(node.var_name)
                    s += '=' + self._generate(node.init_exp) + ';'
                    self.display(node, s)
        
                elif isinstance(node, ast.Pragma):
                    s = '#pragma ' + str(node.pstring) + '\n'
                    self.display(node, s)
                    
                elif isinstance(node, ast.Container):
                    s = self._generate(node.ast)

                elif isinstance(node, ast.DeclStmt):
                    s = self._generate(node.ast)
        
                else:
                    self.display('[module.loop.astvisitors.ExampleVisitor] orio.module.loop.codegen internal error: unrecognized type of AST: %s' % node.__class__.__name__)
            except Exception, e:
                err("[module.loop.astvisitors.ExampleVisitor] Exception in node %s: %s" % (node.__class__.__name__,e))
            
        pass

    def _generate(self, node):
        # Private method
        return self.cgen.generate(node)
    pass    # end of class ExampleVisitor
        
class CountingVisitor(ASTVisitor):
    def __init__(self):
        self.loops = 0
        self.maxnest = 0
        self.adds = 0
        self.mults = 0
        self.divs = 0
        self.lops = 0
        self.reads = 0
        self.writes = 0
        self.comps = 0
        self.gotos = 0        
        self._nest = 0
        self.vars = {}
        pass

    
    def visit(self, nodes, params={}):
        '''Invoke accept method for specified AST node'''
        nodelist = nodes
        if not isinstance(nodes,(list,tuple)):
            nodelist = [nodes]
        for node in nodelist:
            if not node: continue
            try:
                if isinstance(node, ast.NumLitExp):
                    pass
        
                elif isinstance(node, ast.StringLitExp):
                    pass
        
                elif isinstance(node, ast.IdentExp):
                    self.reads += 1
        
                elif isinstance(node, ast.ArrayRefExp):
                    self.visit(node.exp)        # array variable
                    self.visit(node.sub_exp)    # array index
                    
                elif isinstance(node, ast.FunCallExp):
                    self.visit(node.args)
        
                elif isinstance(node, ast.UnaryExp):
                    self.visit(node.exp)  # the operand
                    if node.op_type in [node.PLUS, node.MINUS]:
                        pass
                    elif node.op_type == node.LNOT:
                        self.lops += 1
                    elif node.op_type in [node.PRE_INC, node.PRE_DEC, node.POST_INC, node.POST_DEC]:
                        self.adds += 1
                        self.reads += 1
                        self.writes += 1
                    elif node.op_type == node.DEREF:
                        self.reads += 1
                    elif node.op_type == node.ADDRESSOF:
                        self.reads += 1
                    else:
                        err('[ExampleVisitor] Unknown unary operator:'+ str(node.op_type))
    
                elif isinstance(node, ast.BinOpExp):
                    self.visit(node.lhs)
                    if node.op_type == node.MUL:
                        self.mults += 1
                    elif node.op_type == node.DIV:
                        self.divs += 1
                    elif node.op_type in [node.MOD, node.ADD, node.SUB]:
                        self.adds += 1
                    elif node.op_type in [node.LT, node.GT, node.LE, node.EQ, node.NE]:
                        self.comps += 1
                    elif node.op_type in [node.LOR, node.LAND]:
                        self.lops += 1
                    elif node.op_type == node.COMMA:
                        pass
                    elif node.op_type == node.EQ_ASGN:
                        self.writes += 1
                    else:
                        err('Unknown binary operator:'+ str(node.op_type))
                    self.visit(node.rhs)
        
                elif isinstance(node, ast.ParenthExp):
                    self.visit(node.exp)
        
                elif isinstance(node, ast.Comment):
                    pass
                    
                elif isinstance(node, ast.ExpStmt):
                    self.visit(node.exp)
                    
                elif isinstance(node, ast.GotoStmt):
                    self.gotos += 1
                        
                elif isinstance(node, ast.CompStmt):
                    self.visit(node.stmts)
    
                elif isinstance(node, ast.IfStmt):
                    self.visit(node.test)
                    self.visit(node.true_stmt)
                    self.visit(node.false_stmt)                        
        
                elif isinstance(node, ast.ForStmt):
                    self._nest += 1
                    self.visit(node.init)
                    self.visit(node.test)
                    self.visit(node.iter)
                    self.visit(node.stmt)
                    self.loops += 1
                    if self._nest > self.maxnest: self.maxnest = self._nest    
                    self._nest -= 1
        
                elif isinstance(node, ast.TransformStmt):
                    err('[CountingVisitor] orio.module.loop.codegen internal error: a transformation statement is never generated as an output')
        
                elif isinstance(node, ast.VarDecl):
                    pass
        
                elif isinstance(node, ast.VarDeclInit):
                    self.writes += 1
                    self.visit(node.init_exp)

                elif isinstance(node, ast.Pragma):
                    pass
                    
                elif isinstance(node, ast.Container):
                    self.visit(node.ast)

                elif isinstance(node, ast.DeclStmt):
                    for decl in node.decls:
                        self.visit(decl)
                else:
                    err('[CountingVisitor] orio.module.loop.astvisitors.CountingVisitor internal error: unrecognized type of AST: %s' % node.__class__.__name__)
            except Exception, e:
                err("[CountingVisitor] Exception in node %s: %s" % (node.__class__,e))

            
        pass
        
        
    def __str__(self):
        s = "Code stats:"
        s += '''
        Number of loops: \t%d
        Max loop nest depth: \t%d
        Additions: \t\t%d
        Multiplications: \t%d
        Divisions: \t\t%d
        Logical: \t\t%d
        Reads: \t\t\t%d
        Writes: \t\t%d
        Comparisons:\t\t%d
        Gotos: \t\t\t%d
        ''' % (self.loops,    
            self.maxnest,
            self.adds, 
            self.mults, 
            self.divs, 
            self.lops, 
            self.reads, 
            self.writes, 
            self.comps, 
            self.gotos)
        return s

