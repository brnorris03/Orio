'''
Created on April 26, 2015

@author: norris
'''
from orio.module.loop import ast
from orio.main.util.globals import *
from orio.tool.graphlib import graph
from orio.module.loop import astvisitors

class CFGVertex(graph.Vertex):
    '''A CFG vertex is a basic block.'''
    def __init__(self, name, node=None):
        try: graph.Vertex.__init__(self, name)
        except Exception as e: err("CFGVertex.__init__:" + str(e))
        self.stmts = [node]    # basic block, starting with leader node
        pass
    
    def append(self, node):
        self.stmts.append(node)
        
    def copy(self):
        v = CFGVertex(self.name)
        v.e = self.e
        v.data = self.data
        return v
    
    def succ(self):
        return self.out_v()
    
    def pred(self):
        return self.in_v()
        
    def __str__(self):
        return "<%s> " % self.name + str(self.stmts)
    
    pass  # End of CFG vertex class

class CFGEdge(graph.DirEdge):
    def __init__(self, v1, v2, name=''):
        if not name: name = Globals().incrementCounter()
        graph.DirEdge.__init__(self, name, v1, v2)
        pass
    
    pass # End of CFGEdge class

class CFGGraph(graph.Graph):
    def __init__(self, nodes, name='CFG'):
        graph.Graph.__init__(self, name)
        self.cfgVisitor = CFGVisitor(self)        
        self.cfgVisitor.visit(nodes)

    
    def nodes(self):
        return self.v
    
    def pred(self, bb):
        return self.v[bb.name].in_v()
    
    def succ(self, bb):
        return self.v[bb.name].out_v()
    
    def display(self):
        #sys.stdout.write(str(self))
        self.genDOT()

        
    def genDOT(self, fname=''):
        buf = 'digraph CFG {\n'
        for n,vertex in list(self.v.items()):
            label = '[label="%s%s...",shape=box]' % (n,str(vertex.stmts[0]).split('\n')[0])
            buf += '\t%s %s;\n' % (n, label)
            for edge in vertex.out_e:
                for dv in edge.dest_v:
                    buf += '\t%s -> %s;\n' % (n, dv.name)
        buf += '\n}\n'
        if fname == '': fname = Globals().tempfilename + '.dot'
        f=open(fname,'w')
        f.write(buf)
        f.close()
        
        debug(msg=buf,obj=self,level=4)
        return buf
        

    pass # End of CFG Graph class

class CFGVisitor(astvisitors.ASTVisitor):
    def __init__(self, graph):
        astvisitors.ASTVisitor.__init__(self)
        self.cfg = graph
        v = CFGVertex('_TOP_')
        self.cfg.add_v(v)
        self.stack = [v]
        self.lead = True
        self.verbose = False
        self.last = None

    def display(self, node, msg=''):
        if self.verbose:
            sys.stdout.write("[%s] " % self.__class__.__name__ + node.__class__.__name__ + ': ' + msg+'\n')
        
    def visit(self, nodes, params={}):
        '''Invoke accept method for specified AST node'''
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        try:
            for node in nodes:
                if not node: continue
                v = CFGVertex(node.id, node)
                if isinstance(node, ast.ForStmt):
                    self.display(node)
                    # Children: header: node.init, node.test, node.iter; body: node.stmt
                    v = CFGVertex('ForLoop' + str(node.id), node)
                    self.cfg.add_v(v)
                    self.cfg.add_e(CFGEdge(self.stack.pop(),v))
                    self.stack.append(v)
                    self.lead = True
                    self.stack.append(v)
                    self.visit(node.stmt)

                    vbottom = CFGVertex('_JOIN_' + str(node.id))
                    self.cfg.add_v(vbottom)
                    self.cfg.add_e(CFGEdge(v,vbottom))
                    self.cfg.add_e(CFGEdge(self.stack.pop(),vbottom))
                    self.stack.append(vbottom)
                    self.lead = True
                elif isinstance(node, ast.IfStmt):
                    self.display(node)
                    v = CFGVertex('IfStmt' + str(node.id) , node)
                    self.cfg.add_v(v)
                    self.cfg.add_e(CFGEdge(self.stack.pop(),v))
                    self.stack.append(v)
                    
                    self.lead = True
                    self.visit(node.true_stmt)
                    truelast = self.stack.pop()                    
                    self.stack.append(v)
                    
                    self.lead = True
                    self.visit(node.false_stmt)
                    falselast = self.stack.pop()
                    self.lead = True
                    
                    vbottom = CFGVertex('_JOIN_' + str(node.id))
                    self.cfg.add_v(vbottom)
                    self.cfg.add_e(CFGEdge(truelast,vbottom))
                    self.cfg.add_e(CFGEdge(falselast,vbottom))
                    self.stack.append(vbottom)
                elif isinstance(node, ast.CompStmt):
                    self.display(node)
                    self.visit(node.stmts) 
                # TODO: handle gotos 
                else:
                    # Add to previous basic block
                    if self.lead:
                        v = CFGVertex(node.id, node)
                        self.cfg.add_v(v)
                        self.cfg.add_e(CFGEdge(self.stack.pop(),v))
                        self.stack.append(v)
                        self.lead = False
                    else:
                        self.stack.pop()
                        self.stack.append(v)
                        self.stack[-1].append(node)
        except Exception as ex:
            err("[orio.module.loop.cfg.CFGVisitor.visit()] %s" % str(ex))
                                   
        return

    def getCFG(self):
        return self.cfg
    
        
    pass   # end of class CFGVisitor

