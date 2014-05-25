'''
Created on Mar 8, 2012

@author: norris
'''

from orio.module.matrix.elements import *

class Visitor:
    '''AST visitor interface
    '''
    def __init__(self):
        self.repository = []
        self.seen = []
        self.reverseEdges = 0
        self.vertexNumber = 0
        pass
  
    def getRepository(self):
        '''
        Retrieve auxilliary ASTs
        '''
        return self.repository
  
    def setRepository(self, repository):
        '''
        Set auxilliary ASTs
        '''
        self.repository = repository
  
    def visitVertex(self, vertex):
        self.discoverVertex(vertex)
        self.traverseVertex(vertex)
        self.finishVertex(vertex)
  
    def visitStatement(self, node):
        self.visitVertex(node)
  
    def visitType(self, node):
        self.visitVertex(node)
  
    def visitTypeUniverse(self, node):
        self.visitVertex(node)
  
    def visitEnumeration(self, node):
        self.visitVertex(node)
  
    def visitEnumerator(self, node):
        self.visitVertex(node)
  
    def visitArray(self, node):
        self.visitVertex(node)

    def visitVector(self, node):
        self.visitVertex(node)

    def visitMatrix(self, node):
        self.visitVertex(node)
    
    def visitIterator(self, node):
        self.visitVertex(node)
    
    def visitParameter(self, node):
        self.visitVertex(node)
    
    def visitMethod(self, node):
        self.visitVertex(node)
    
    def getReverseEdges(self):
        '''
        Retrieve the flag for reversing the orientation of edges in the traversal
        '''
        return self.reverseEdges
    
    def setReverseEdges(self, reverseEdges):
        '''
        Set the flag for reversing the orientation of edges in the traversal
        '''
        self.reverseEdges = reverseEdges
    
    def traverseVertex(self, vertex):
        '''
        Traverse each edge of this vertex
        '''
        pass
    
    def resetVertexNumbering(self):
        '''
        Reset the vertex discovery and finish numbering
        '''
        self.vertexNumber = 0
    
    def discoverVertex(self, vertex):
        '''
        Mark a vertex as discovered and increment the count
        '''
        self.vertexNumber += 1
    
    def finishVertex(self, vertex):
        '''
        Mark a vertex as finished and increment the count
        '''
        self.vertexNumber += 1
    
    def getVerticesSeen(self):
        '''
        Retrieve the list of vertices discovered
        '''
        return self.seen
    
    def setVerticesSeen(self, seen):
        '''
        Set the list of vertices discovered
        '''
        self.seen = list(seen)
    
    def getVertexSeen(self, vertex):
        '''
        Retrieve the vertex discovery state
        '''
        return vertex in self.seen
    
    def setVertexSeen(self, vertex, state):
        '''
        Set the vertex discovery state
        '''
        if state:
            if not (vertex in self.seen):
                self.seen.append(vertex)
        else:
            if vertex in self.seen:
                self.seen.remove(vertex)
    # Ending class methods section
    pass
