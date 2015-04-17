'''
Created on Feb 26, 2015

@author: norris
'''
import ast, sys, os
import orio.main.util.globals as g
import orio.module.loop.codegen

class ASTVisitor:
    ''' Standard visitor pattern abstract class'''
    def __init__(self):
        pass
    
    def __str__(self):
        return self.__class__.__name__
    
    def visit(self, node, params={}):
        ''' Default visit method to be overwritten by specific visitors'''
        pass

class ExampleVisitor(ASTVisitor):
    def __init__(self):
        self.cgen = orio.module.loop.codegen.CodeGen_C()
        if 'ORIO_TEST' in os.environ.keys():
            self.verbose = True
        else:
            self.verbose = False
        
    def display(self,msg):
        if self.verbose:
            sys.stdout.write(msg+'\n')
        
    def visit(self, nodes, preorder=True, params={}):
        '''Invoke accept method for specified AST node'''

        for node in nodes:
            try:
                if isinstance(node, ast.NumLitExp):
                    self.display("[ExampleVisitor] Visiting NumLitExp: %s" % str(node.val))
        
                elif isinstance(node, ast.StringLitExp):
                    self.display("[ExampleVisitor] Visiting StringLitExp: %s" % str(node.val))
        
                elif isinstance(node, ast.IdentExp):
                    self.display("[ExampleVisitor] Visiting IdentExp: %s" % str(node.name))
        
                elif isinstance(node, ast.ArrayRefExp):
                    self.display("Visting ArrayRefExp: %s" % self._generate(node.exp))
                    
                elif isinstance(node, ast.FunCallExp):
                    s = self._generate(node.exp) + '('
                    s += ','.join(map(lambda x: self._generate(x), node.args))
                    s += ')'
                    self.display("[ExampleVisitor] Visiting FunCallExp: %s" % s)
        
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
                    self.display('[ExampleVisitor] Visiting UnaryExp: %s' % s)
    
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
                    self.display('[ExampleVisitor] Visiting BinaryExp: %s' % s)
        
                elif isinstance(node, ast.ParenthExp):
                    s = '(' + self._generate(node.exp) + ')'
                    self.display('[ExampleVisitor] Visiting ParenthExp: %s' % s)
        
                elif isinstance(node, ast.Comment):
                    s = '/*' + str(node.text) + '*/'
                    self.display('[ExampleVisitor] Visiting Comment: %s' % s)
                    
                elif isinstance(node, ast.ExpStmt):
                    s=''
                    if node.getLabel(): s += str(node.getLabel()) + ':'
                    if node.exp:
                        s += str(self._generate(node.exp))
                    self.display('[ExampleVisitor] Visiting ExpStmt: %s' % s)
        
                elif isinstance(node, ast.GotoStmt):
                    s = ''
                    if node.getLabel(): s += node.getLabel() + ':'
                    if node.target:
                        s += 'goto ' + str(node.target) + ';\n'
                    self.display('[ExampleVisitor] Visiting GotoStmt: %s' % s)
                        
                elif isinstance(node, ast.CompStmt):
                    self.alldecls = set([])
                    self.display('[ExampleVisitor] Visiting CompStmt')
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
                    self.display('[ExampleVisitor] Visiting IfStmt:\n%s' % s)
                            
        
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
                    self.display('[ExampleVisitor] Visiting ForStmt:\n%s' % s)
    
        
                elif isinstance(node, ast.TransformStmt):
                    self.display('[ExampleVisitor] orio.module.loop.codegen internal error: a transformation statement is never generated as an output')
        
                elif isinstance(node, ast.VarDecl):
                    s =''
                    sv = str(node.type_name) + ' '
                    sv += ', '.join(node.var_names)
                    sv += ';\n'
                    if not sv in self.alldecls: 
                        s += sv
                        self.alldecls.add(sv)
                    self.display('[ExampleVisitor] Visiting VarDecl: %s' % s)
        
                elif isinstance(node, ast.VarDeclInit):
                    s = str(node.type_name) + ' '
                    s += self._generate(node.var_name)
                    s += '=' + self._generate(node.init_exp) + ';'
                    self.display('[ExampleVisitor] Visiting VarDeclInit: %s' % s)
        
                elif isinstance(node, ast.Pragma):
                    s = '#pragma ' + str(node.pstring) + '\n'
                    self.display('[ExampleVisitor] Visiting Pragma: %s' % s)
                    
                elif isinstance(node, ast.Container):
                    s = self._generate(node.ast)
        
                else:
                    self.display('[ExampleVisitor] orio.module.loop.codegen internal error: unrecognized type of AST: %s' % node.__class__.__name__)
            except Exception, e:
                print "Exception in node %s: %s" % (node.__class__,e)
                import traceback
                traceback.print_stack()
                raise e
            
            pass

    def _generate(self, node):
        # Private method
        return self.cgen.generate(node)
        
