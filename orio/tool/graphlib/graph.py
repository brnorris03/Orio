#!/usr/bin/env python

'''Directed and undirected graph data structures and algorithms.

Copyright 2005, Robert Dick (dickrp@ece.northwestern.edu).  Numerous bug fixes
and improvements from Kosta Gaitanis (gaitanis@tele.ucl.ac.be).  Please see the
license file for legal information.'''

'''Modified for backward compatibility with Python 2.3 by Boyana Norris
(norris@mcs.anl.gov).'''

__version__ = '0.4.1'
__author__ = 'Robert Dick and Kosta Gaitanis'
__author_email__ = 'dickrp@ece.northwestern.edu and gaitanis@tele.ucl.ac.be'

import struct
from orio.tool.graphlib import delegate

# The following four lines borrowed from Gregory R. Warnes's fpconst
# (until it's standard in Python distributions).
if struct.pack('i',1)[0] != '\x01':
    PosInf = struct.unpack('d', '\x7F\xF0\x00\x00\x00\x00\x00\x00')[0]
else:
    PosInf = struct.unpack('d', '\x00\x00\x00\x00\x00\x00\xf0\x7f')[0]


class GraphError(Exception):
    '''Exception for graph errors.'''
    pass


class SortStruct(float, object):
    '''Throw-away class for temporarily holding values.

    Initialize with a float distance and as many keyword args as desired.
    The keyword args become attributes.'''

    def __new__(cls, val, **kargs):
        obj = float.__new__(cls, val)
        obj.__dict__ = kargs
        return obj

    #@property
    def number(self):
        return float(self)
    number = property(number)
  

def _condop(cond, v1, v2):
    if cond:
        return v1
    else:
        return v2

def _roprop(description = None):
    def prop_func(ro_method):
        return property(ro_method, None, None, description)
    return prop_func

#================================================================================

#class Vertex(delegate.Delegate):
class Vertex:
    '''General graph vertex.

    All methods colaborate with directed and undirected
    edges.  If the edges are undirected, ine == oute.'''

    def attach_e(self, e):
        '''Attach an edge.'''
        self.e.append(e)

    def connecting_e(self, v):
        '''List of edges connecting self and other vertex.'''
        return [e for e in self.e if e.leaves(self) and e.enters(v)]

    def __init__(self, name):
        '''Name neen't be a string but must be hashable and immutable.'''
        #self.__Delegate.__init__(self)
        self.e = []
        self.name = name
        self.data = {}
        return

    def __str__(self): return str(self.name)
    
#    def __getstate__(self):
#        '''Need to break cycles to prevent recursion blowup in pickle.
#        
#        Dump everything except for edges.'''
#        dcp = copy.copy(self.__dict__)
#        dcp['_e'] = []
#        return dcp

    #@_roprop('All edges.')
    def all_e(self): return self.e
    all_e = _roprop('All edges.')(all_e)

    #@_roprop('Incoming edges.')
    def in_e(self): return [e for e in self.e if e.enters(self)]
    in_e = _roprop('Incoming edges.')(in_e)

    #@_roprop('Outgoing edges.')
    def out_e(self): return [e for e in self.e if e.leaves(self)]
    out_e = _roprop('Outgoing edges.')(out_e)

    #@_roprop('Set of adjacent vertices.  Edge direction ignored.')
    def adjacent_v(self):
        adj = set([v for e in self.e for v in e.all_v])
        adj.remove(self)
        return adj
    adjacent_v = _roprop('Set of adjacent vertices.  Edge direction ignored.')(adjacent_v)

    #@_roprop('Set of vertices connected by incoming edges.')
    def in_v(self):
        return set([v for e in self.e for v in e.src_v if e.enters(self)])
    in_v = _roprop('Set of vertices connected by incoming edges.')(in_v)
    
    def in_v_names(self):
        return [v for e in self.e for v in e.src_v if e.enters(self)]
    
    #@_roprop('Set of vertices connected by outgoing edges.')
    def out_v(self):
        return set([v for e in self.e for v in e.dest_v if e.leaves(self)])
    out_v = _roprop('Set of vertices connected by outgoing edges.')(out_v)

    #@_roprop('True if vertex has no outgoing edges.')
    def is_sink(self): return not self.out_e
    is_sink = _roprop('True if vertex has no outgoing edges.')(is_sink)

    #@_roprop('True if vertex has no incoming edges.')
    def is_src(self): return not self.in_e
    is_src = _roprop('True if vertex has no incoming edges.')(is_src)

    #@_roprop('True if vertex has incoming and outgoing edges.')
    def is_intermed(self): return not self.is_sink and not self.is_src
    is_intermed = _roprop('True if vertex has incoming and outgoing edges.')(is_intermed)


class RawEdge(delegate.Delegate):
    '''Base class for undirected and directed edges.  Not directly useful.'''

    def __init__(self, name, v1, v2):
        '''Name needn't be a string but must be immutable and hashable.'''
        self.__Delegate.__init__(self)
        self.name = name
        self.v = [v1, v2]
        v1.attach_e(self)
        v2.attach_e(self)

#    def __setstate__(self, state):
#        '''Restore own state and add self to connected vertex edge lists.'''
#        self.__dict__ = state
#        self._v[0].attach_e(self)
#        self._v[1].attach_e(self)

    #@_roprop('All connected vertices.')
    def all_v(self): return self.v
    all_v = _roprop('All connected vertices.')(all_v)


class UndirEdge(RawEdge):
    '''Undirected edge.'''

    def enters(self, v):
        '''True if this edge is connected to the vertex.'''
        return v in self.v

    def leaves(self, v):
        '''True if this edge is connected to the vertex.'''
        return v in self.v

    def __str__(self):
        return '%s: %s -- %s' % (str(self.name), str(self.v[0]),
            str(self.v[1]))

    def weight(self, v1, v2):
        '''1 if this edge connects the vertices.'''
        if v1 not in self.v or v2 not in self.v:
            raise GraphError('vertices not connected')
        return 1        

    #@_roprop('Source vertices.')
    def src_v(self): return self.v
    src_v = _roprop('Source vertices.')(src_v)

    #@_roprop('Destination vertices.')
    def dest_v(self): return self.v
    dest_v = _roprop('Destination vertices.')(dest_v)


class DirEdge(RawEdge):
    '''Directed edge.'''

    def enters(self, v):
        '''True only if the vertex is this edge's destination.'''
        return v is self.v[1]

    def leaves(self, v):
        '''True only if the vertex is this edge's source.'''
        return v is self.v[0]

    def invert(self):
        '''Inverts the edge direction.'''
        self.v.reverse()

    def weight(self, v1, v2):
        '''1 if this edge has v1 as its source and v2 as its destination.'''
        if v1 is not self.v[0] or v2 is not self.v[1]:
            raise GraphError('vertices not connected')
        return 1

    def __str__(self):
        return '%s: %s -> %s' % (str(self.name), str(self.v[0]),
            str(self.v[1]))

    #@_roprop(
    #    '''Single element list containing the edge's source vertex.
    #    
    #    Must be a list to conform with UndirEdge's interface.''')
    def src_v(self): return [self.v[0]]
    src_v = _roprop(
        '''Single element list containing the edge's source vertex.
        
        Must be a list to conform with UndirEdge's interface.''')(src_v)

    #@_roprop(
    #    '''Single element list containing the edge's destination vertex.
    #
    #    Must be a list to conform with UndirEdge's interface.''')
    def dest_v(self): return [self.v[1]]
    dest_v = _roprop(
        '''Single element list containing the edge's destination vertex.

        Must be a list to conform with UndirEdge's interface.''')(dest_v)

#================================================================================

def DataWrapped(cls):
    '''Returns a wrapped class with a new 'data' member.'''

    class __Wr(cls):
        '''Wrapper to add 'data' to another class.'''

        def __init__(self, *pargs, **kargs):
            '''Use last parg for data and sends all other args to base.'''
            if kargs:
                cls.__init__(self, *pargs[:-1], **kargs)
            else:
                cls.__init__(self, *pargs[:-1])
            self.data = pargs[-1]

        def __str__(self):
            '''Append data to base's __str__.'''
            return ' '.join([_f for _f in (cls.__str__(self), str(self.data)) if _f])

    __Wr.__name__ = 'Data%s' % cls.__name__.split('.')[-1]
    return __Wr

DataVertex = DataWrapped(Vertex)
DataUndirEdge = DataWrapped(UndirEdge)
DataDirEdge = DataWrapped(DirEdge)

#================================================================================

class VertexDict(dict, delegate.Delegate):
    '''Dictionary of vertices.'''

    def __init__(self, graph):
        self.__dict__.__init__(self)
        self.__Delegate.__init__(self)
        self.graph = graph

    def __setitem__(self, key, val):
        if key in self:
            raise KeyError('VertexDict already has (%s, %s).' % (key, val))
        self.__dict.__setitem__(self, key, val)

    def __delitem__(self, key):
        '''Delete all edges connected to the vertex, along with the vertex.'''
        v = self[key]
        for e in v.e:
            del self.graph.e[e.name]
        self.__dict.__delitem__(self, key)
    
    def deleteItem(self, key):
        dict.__delitem__(self,key)


class EdgeDict(dict, delegate.Delegate):
    '''Dictionary of edges.'''

    def __setitem__(self, key, val):
#        if self.has_key(key):
#            raise KeyError('EdgeDict already has (%s, %s).' % (key, val))
        self.__dict.__setitem__(self, key, val)

    def __delitem__(self, key):
        '''Remove edge from vertices to which it is attached.'''
        e = self[key]
        for v in e.v:
            try: v.e.remove(e)
            except: pass
        self.__dict.__delitem__(self, key)

    def deleteItem(self, key):
        dict.__delitem__(self,key)

#================================================================================
# Introduce an equivalent for the Python2.4 sorted builtin
def bn_sorted(dict):
    skeys = list(dict.keys())
    skeys.sort()
    slist = []
    for k in skeys:
        slist.append(dict[k])
    return slist
     
# Graph class definition
class Graph(delegate.Delegate):
    '''General-purpose graph data structure.'''

    def __init__(self, name = None):
        self.name = name
        self.v = VertexDict(self)
        self.e = EdgeDict()
 
    def __str__(self): 
        return self.__class__.__name__ + \
            _condop(self.name, ' ' + str(self.name), '') + '\nVertices:\n' + \
            '\n'.join([str(v) for v in bn_sorted(self.v)]) + \
            '\nEdges:\n' + \
            '\n'.join([str(e) for e in bn_sorted(self.e)]) + '\n'
    
    def clear(self):
        self.name = None
        self.v.clear()
        self.e.clear()

    #@_roprop('List of all vertices without incoming edges.')
    def src_v(self): return [v for n, v in list(self.v.items()) if v.is_src]
    src_v = _roprop('List of all vertices without incoming edges.')(src_v)

    #@_roprop('List of all vertices without outgoing edges.')
    def sink_v(self): return [v for n, v in list(self.v.items()) if v.is_sink]
    sink_v = _roprop('List of all vertices without outgoing edges.')(sink_v)

    #@_roprop(
    #    'List of all vertices with both incoming and outgoing edges.')
    def intermed_v(self):    return [v for n, v in list(self.v.items()) if v.is_intermed]
    intermed_v = _roprop(
        'List of all vertices with both incoming and outgoing edges.')(intermed_v)

    def add_v(self, v):
        '''Add and return a vertex.'''
        self.v[v.name] = v
        v.g = self
        return v

    def add_e(self, e):
        '''Add and return an edge.'''
        self.e[e.name] = e
        return e
        
    def connected_components(self):
        '''Return a list of lists.  Each holds transitively-connected vertices.'''
        unchecked = set(list(self.v.values()))
        groups = []
        while len(unchecked):
            vcon = self.depth_first_search(unchecked.pop())
            unchecked -= set(vcon)
            groups.append(vcon)
        return groups

    #@staticmethod
    def depth_first_search(start_v, visitor = None, reverse=False):
        '''Return a depth-first search list of vertices.'''
        unprocessed = [start_v]
        visited = []
        while unprocessed:
            v = unprocessed.pop()
            if v not in visited:
                if visitor is not None and callable(visitor):
                    v.visitor()
                visited.append(v)
                if reverse: unprocessed.extend(v.in_v)
                else: unprocessed.extend(v.out_v)
        return visited
    depth_first_search = staticmethod(depth_first_search)

    #@staticmethod
    def breadth_first_search(start_v, ignore_action='', reverse=False):
        '''Return a breadth-first search list of vertices.'''
        unprocessed = [start_v]
        visited = []
        while unprocessed:
            v = unprocessed.pop(0)
            if v not in visited:
                visited.append(v)
                if reverse: unprocessed.extend(v.in_v)
                else: unprocessed.extend(v.out_v)
        return visited
    breadth_first_search = staticmethod(breadth_first_search)

    #@staticmethod
    def topological_sort(start_v):
        '''Return a topological sort list of vertices.'''
        unprocessed = [start_v]
        visited = []
        while unprocessed:
            v = unprocessed.pop(0)
            incoming_v = v.adjacent_v - v.out_v
            if v not in visited and not (incoming_v - set(visited)):
                visited.append(v)
                unprocessed.extend(v.out_v)
        return visited
    topological_sort = staticmethod(topological_sort)

#================================================================================

    def minimal_span_tree(self, **kargs):
        '''Return minimal spanning 'Tree'.

        Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
        returning a weight.  Defaults to e.weight().
        'targets' list of target vertices.  Defaults to all vertices.'''

        def def_weight_func(e, v1, v2): return e.weight(v1, v2)
        weight_func = kargs.get('weight_func', def_weight_func)
        targets = kargs.get('targets', list(self.v.values()))
        visited = set([targets.pop()])
        unvisited = set(list(self.v.values())) - visited
        mst = Tree()

        while targets:
# Haven't found it yet.  Search more.
            connected = []
            for v in visited:
                for u in unvisited:
                    conn = v.connecting_e(u)
                    if conn:
                        dist = weight_func(conn.pop(), v, u)
                        connected.append(SortStruct(dist, src=v, dest=u))

            if not connected:
                raise GraphError('unreachable vertices in minimal_spanning_tree')

# Connect it
            near = min(connected)
            if near.src.name not in mst.v:
                mst.add_v(DataVertex(near.src.name, near.src))
            if near.dest.name not in mst.v:
                mst.add_v(DataVertex(near.dest.name, near.src))
            e = mst.auto_add_e(DirEdge(len(mst.e), mst.v[near.src.name],
                mst.v[near.dest.name]))

            visited.add(near.dest)
            unvisited.remove(near.dest)
            targets.remove(near.dest)
        return mst

#================================================================================

    def shortest_tree(self, start, **kargs):
        '''Return a 'Tree' of shortest paths to all nodes.

        Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
        returning a weight.  Defaults to e.weight().
        'targets' list of target vertices.  Defaults to all vertices.'''

        def def_weight_func(e, v1, v2): return e.weight(v1, v2)
        weight_func = kargs.get('weight_func', def_weight_func)
        targets = set(kargs.get('targets', list(self.v.values())))
        path_tr = Tree()
        dist = {start:0.0}
        unvisited = set(list(self.v.values()))
        while targets:
# Determine the closest vertex
            closest = min([(dist.get(v, PosInf), v) for v in unvisited])[1]
# Add it and push the distances
            unvisited.remove(closest)
            targets.discard(closest)
            adj_v = closest.out_v
            if not adj_v - unvisited:
                raise GraphError('unreachable vertices in shortest_tree')
            for v in adj_v:
                for e in closest.connecting_e(v):
                    push_dist = weight_func(e, closest, v) + dist[closest]
                    if push_dist < dist.get(v, PosInf):
                        dist[v] = push_dist
# Add the vertices.  Remove the old edges, if any.  Add the new edge.
                        if closest.name not in path_tr.v:
                            path_tr.add_v(DataVertex(closest.name, closest))
                        if v.name not in path_tr.v:
                            path_tr.add_v(DataVertex(v.name, v))
                        e_nm = len(path_tr.e)
                        for e2 in path_tr.v[v.name].in_e:
                            del path_tr.e[e2.name]
                            e_nm = e2.name
                        path_tr.auto_add_e(DirEdge(e_nm, path_tr.v[closest.name],
                            path_tr.v[v.name]))
        return path_tr

#================================================================================

    def greedy_paths(self, start, goal, weight_func = None):
        '''Return a dict of greedy paths with (start vertex, end vertex) keys.

        Always makes the highest-gain decision.  Will find a path if one exists.
        Not necessarily optimal.  'weight_func' is a function of (edge, v1, v2)
        returning a weight.  Defaults to e.weight()'''

        def def_weight_func(e, v1, v2): return e.weight(v1, v2)
        weight_func = weight_func or def_weight_func
        path = [start]
        visited = set([start])
        while path[-1] is not goal:
            adj_v = [SortStruct(weight_func(e, path[-1], v), dest=v)
                for e in path[-1].out_e for v in e.dest_v if v not in visited]
            if adj_v:
                closest_v = min(adj_v).dest
                visited.add(closest_v)
                path.append(closest_v)
            else:
                path.pop()
# Prepare the dict
        d = {}
        for i1 in range(len(path)):
            for i2 in range(i1, len(path)):
                d[path[i1], path[i2]] = path[i1:i2 + 1]
        return d

    def all_pairs_sp(self, weight_func = None):
        '''Return a dictionary of shortest path lists for all vertex pairs.

        Keys are (source, destination) tuples.
        'weight_func' is a function taking (edge, v1, v2) that returns a weight.
        Defaults to e.weight()'''

        return dict([list(self.shortest_tree(v,
            weight_func=weight_func).path_dict().items()) for v in list(self.v.values())])

#================================================================================

    #@staticmethod
    def path_weight(path, weight_func = None):
        '''Return the weight of the path, which is a list of vertices.

        'weight_func' is a function taking (edge, v1, v2) and returning a weight.'''

        def def_weight_func(e, v1, v2): return e.weight(v1, v2)
        weight_func = weight_func or def_weight_func
        wt = 0.0
        for v1, v2 in zip(path[:-1], path[1:]):
            connect_e = [e for e in v1.out_e if e.enters(v2)]
            if not connect_e:
                raise GraphError('vertices in path are not connected')
            wt += min([weight_func(e, v1, v2) for e in connect_e])
        return wt
    path_weight = staticmethod(path_weight)

#================================================================================

class TreeError(GraphError): pass

class Tree(Graph):
    '''Tree data structure.

    Edges must be directed and reconvergent paths do not occur.'''

    def add_e(self, e):
        '''Add edge if tree invariant holds.  Otherwise, raise exception.'''
        if len(e.dest_v) != 1:
            raise TreeError('undirected edge')
        if len(e.dest_v[0].in_e) != 1:
            raise TreeError('edge introduces reconvergent paths into tree')
        return self.__Graph.add_e(self, e)

    def auto_add_e(self, e):
        '''Automatically choose correct direction for new edge.'''
        if len(e.dest_v) != 1:
            raise TreeError('undirected edge')
        if len(e.dest_v[0].in_e) != 1:
            e.invert()
        self.add_e(e)

    def is_safe_e(self, target_v):
        '''True if edge maintains tree invariant.'''
        return not target_v.in_e

    def path_dict(self):
        '''Return a dictionary of path lists from the root to each vertex.

        Keys are (source vertex, destination vertex) tuples.'''

        root = self.src_v
        assert len(root) == 1
        root = root.pop()

        d = {(root.data, root.data):[root.data]}
        for v in self.depth_first_search(root):
            parent = v.in_v
            assert len(parent) <= 1
            if parent:
                parent = parent.pop()
                d[root.data, v.data] = d[root.data, parent.data] + [v.data]
        return d

#################################################################################
if __name__ == '__main__':
    G = Graph()
    a, b, c, d, e, f, g = [G.add_v(Vertex(nm)) for nm in 'a b c d e f g'.split()]
    for ep in [(a, b), (a, c), (b, d), (b, f), (b, e), (c, e), (d, f), (e, f),
    (f, g)]:
        G.add_e(UndirEdge(len(G.e), *ep))

    print(G)
    print('DFS:', list(map(str, G.depth_first_search(a))))
    print('BFS:', list(map(str, G.breadth_first_search(a))))
    print('top sort:', list(map(str, G.topological_sort(a))))

    T = G.minimal_span_tree()
    print(T)
    print([(list(map(str, k)), list(map(str, v))) for k, v in list(T.path_dict().items())])
    
    S = G.shortest_tree(a)
    print(S)
