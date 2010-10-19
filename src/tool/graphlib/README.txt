Help on module graph:

NAME
    graph - Directed and undirected graph data structures and algorithms.

FILE
    /home/dickrp/svn/support/python/graph/graph.py

DESCRIPTION
    Copyright 2005, Robert Dick (dickrp@ece.northwestern.edu).  Numerous bug fixes
    and improvements from Kosta Gaitanis (gaitanis@tele.ucl.ac.be).  Please see the
    license file for legal information.

CLASSES
    __builtin__.dict(__builtin__.object)
        EdgeDict(__builtin__.dict, delegate.Delegate)
        VertexDict(__builtin__.dict, delegate.Delegate)
    __builtin__.float(__builtin__.object)
        SortStruct(__builtin__.float, __builtin__.object)
    __builtin__.object
        SortStruct(__builtin__.float, __builtin__.object)
    delegate.Delegate(__builtin__.object)
        EdgeDict(__builtin__.dict, delegate.Delegate)
        Graph
            Tree
        RawEdge
            DirEdge
                DataDirEdge
            UndirEdge
                DataUndirEdge
        Vertex
            DataVertex
        VertexDict(__builtin__.dict, delegate.Delegate)
    exceptions.StandardError(exceptions.Exception)
        GraphError
            TreeError
    
    class DataDirEdge(DirEdge)
     |  Wrapper to add 'data' to another class.
     |  
     |  Method resolution order:
     |      DataDirEdge
     |      DirEdge
     |      RawEdge
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, *pargs, **kargs)
     |      Use last parg for data and sends all other args to base.
     |  
     |  __str__(self)
     |      Append data to base's __str__.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DirEdge:
     |  
     |  enters(self, v)
     |      True only if the vertex is this edge's destination.
     |  
     |  invert(self)
     |      Inverts the edge direction.
     |  
     |  leaves(self, v)
     |      True only if the vertex is this edge's source.
     |  
     |  weight(self, v1, v2)
     |      1 if this edge has v1 as its source and v2 as its destination.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from DirEdge:
     |  
     |  dest_v
     |      Single element list containing the edge's destination vertex.
     |      
     |      Must be a list to conform with UndirEdge's interface.
     |  
     |      <get> = dest_v(self)
     |  
     |  src_v
     |      Single element list containing the edge's source vertex.
     |      
     |      Must be a list to conform with UndirEdge's interface.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawEdge:
     |  
     |  __setstate__(self, state)
     |      Restore own state and add self to connected vertex edge lists.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from RawEdge:
     |  
     |  all_v
     |      All connected vertices.
     |  
     |      <get> = all_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class DataUndirEdge(UndirEdge)
     |  Wrapper to add 'data' to another class.
     |  
     |  Method resolution order:
     |      DataUndirEdge
     |      UndirEdge
     |      RawEdge
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, *pargs, **kargs)
     |      Use last parg for data and sends all other args to base.
     |  
     |  __str__(self)
     |      Append data to base's __str__.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from UndirEdge:
     |  
     |  enters(self, v)
     |      True if this edge is connected to the vertex.
     |  
     |  leaves(self, v)
     |      True if this edge is connected to the vertex.
     |  
     |  weight(self, v1, v2)
     |      1 if this edge connects the vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from UndirEdge:
     |  
     |  dest_v
     |      Destination vertices.
     |  
     |      <get> = dest_v(self)
     |  
     |  src_v
     |      Source vertices.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawEdge:
     |  
     |  __setstate__(self, state)
     |      Restore own state and add self to connected vertex edge lists.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from RawEdge:
     |  
     |  all_v
     |      All connected vertices.
     |  
     |      <get> = all_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class DataVertex(Vertex)
     |  Wrapper to add 'data' to another class.
     |  
     |  Method resolution order:
     |      DataVertex
     |      Vertex
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, *pargs, **kargs)
     |      Use last parg for data and sends all other args to base.
     |  
     |  __str__(self)
     |      Append data to base's __str__.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Vertex:
     |  
     |  __getstate__(self)
     |      Need to break cycles to prevent recursion blowup in pickle.
     |      
     |      Dump everything except for edges.
     |  
     |  attach_e(self, e)
     |      Attach an edge.
     |  
     |  connecting_e(self, v)
     |      List of edges connecting self and other vertex.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from Vertex:
     |  
     |  adjacent_v
     |      Set of adjacent vertices.  Edge direction ignored.
     |  
     |      <get> = adjacent_v(self)
     |  
     |  all_e
     |      All edges.
     |  
     |      <get> = all_e(self)
     |  
     |  in_e
     |      Incoming edges.
     |  
     |      <get> = in_e(self)
     |  
     |  in_v
     |      Set of vertices connected by incoming edges.
     |  
     |      <get> = in_v(self)
     |  
     |  is_intermed
     |      True if vertex has incoming and outgoing edges.
     |  
     |      <get> = is_intermed(self)
     |  
     |  is_sink
     |      True if vertex has no outgoing edges.
     |  
     |      <get> = is_sink(self)
     |  
     |  is_src
     |      True if vertex has no incoming edges.
     |  
     |      <get> = is_src(self)
     |  
     |  out_e
     |      Outgoing edges.
     |  
     |      <get> = out_e(self)
     |  
     |  out_v
     |      Set of vertices connected by outgoing edges.
     |  
     |      <get> = out_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class DirEdge(RawEdge)
     |  Directed edge.
     |  
     |  Method resolution order:
     |      DirEdge
     |      RawEdge
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __str__(self)
     |  
     |  enters(self, v)
     |      True only if the vertex is this edge's destination.
     |  
     |  invert(self)
     |      Inverts the edge direction.
     |  
     |  leaves(self, v)
     |      True only if the vertex is this edge's source.
     |  
     |  weight(self, v1, v2)
     |      1 if this edge has v1 as its source and v2 as its destination.
     |  
     |  ----------------------------------------------------------------------
     |  Properties defined here:
     |  
     |  dest_v
     |      Single element list containing the edge's destination vertex.
     |      
     |      Must be a list to conform with UndirEdge's interface.
     |  
     |      <get> = dest_v(self)
     |  
     |  src_v
     |      Single element list containing the edge's source vertex.
     |      
     |      Must be a list to conform with UndirEdge's interface.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawEdge:
     |  
     |  __init__(self, name, v1, v2)
     |      Name needn't be a string but must be immutable and hashable.
     |  
     |  __setstate__(self, state)
     |      Restore own state and add self to connected vertex edge lists.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from RawEdge:
     |  
     |  all_v
     |      All connected vertices.
     |  
     |      <get> = all_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class EdgeDict(__builtin__.dict, delegate.Delegate)
     |  Dictionary of edges.
     |  
     |  Method resolution order:
     |      EdgeDict
     |      __builtin__.dict
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __delitem__(self, key)
     |      Remove edge from vertices to which it is attached.
     |  
     |  __setitem__(self, key, val)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__ = <attribute '__weakref__' of 'EdgeDict' objects>
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from __builtin__.dict:
     |  
     |  __cmp__(...)
     |      x.__cmp__(y) <==> cmp(x,y)
     |  
     |  __contains__(...)
     |      D.__contains__(k) -> True if D has a key k, else False
     |  
     |  __eq__(...)
     |      x.__eq__(y) <==> x==y
     |  
     |  __ge__(...)
     |      x.__ge__(y) <==> x>=y
     |  
     |  __getattribute__(...)
     |      x.__getattribute__('name') <==> x.name
     |  
     |  __getitem__(...)
     |      x.__getitem__(y) <==> x[y]
     |  
     |  __gt__(...)
     |      x.__gt__(y) <==> x>y
     |  
     |  __hash__(...)
     |      x.__hash__() <==> hash(x)
     |  
     |  __init__(...)
     |      x.__init__(...) initializes x; see x.__class__.__doc__ for signature
     |  
     |  __iter__(...)
     |      x.__iter__() <==> iter(x)
     |  
     |  __le__(...)
     |      x.__le__(y) <==> x<=y
     |  
     |  __len__(...)
     |      x.__len__() <==> len(x)
     |  
     |  __lt__(...)
     |      x.__lt__(y) <==> x<y
     |  
     |  __ne__(...)
     |      x.__ne__(y) <==> x!=y
     |  
     |  __repr__(...)
     |      x.__repr__() <==> repr(x)
     |  
     |  clear(...)
     |      D.clear() -> None.  Remove all items from D.
     |  
     |  copy(...)
     |      D.copy() -> a shallow copy of D
     |  
     |  get(...)
     |      D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
     |  
     |  has_key(...)
     |      D.has_key(k) -> True if D has a key k, else False
     |  
     |  items(...)
     |      D.items() -> list of D's (key, value) pairs, as 2-tuples
     |  
     |  iteritems(...)
     |      D.iteritems() -> an iterator over the (key, value) items of D
     |  
     |  iterkeys(...)
     |      D.iterkeys() -> an iterator over the keys of D
     |  
     |  itervalues(...)
     |      D.itervalues() -> an iterator over the values of D
     |  
     |  keys(...)
     |      D.keys() -> list of D's keys
     |  
     |  pop(...)
     |      D.pop(k[,d]) -> v, remove specified key and return the corresponding value
     |      If key is not found, d is returned if given, otherwise KeyError is raised
     |  
     |  popitem(...)
     |      D.popitem() -> (k, v), remove and return some (key, value) pair as a
     |      2-tuple; but raise KeyError if D is empty
     |  
     |  setdefault(...)
     |      D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D
     |  
     |  update(...)
     |      D.update(E, **F) -> None.  Update D from E and F: for k in E: D[k] = E[k]
     |      (if E has keys else: for (k, v) in E: D[k] = v) then: for k in F: D[k] = F[k]
     |  
     |  values(...)
     |      D.values() -> list of D's values
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from __builtin__.dict:
     |  
     |  __new__ = <built-in method __new__ of type object>
     |      T.__new__(S, ...) -> a new object with type S, a subtype of T
     |  
     |  fromkeys = <built-in method fromkeys of _delegate_meta object>
     |      dict.fromkeys(S[,v]) -> New dict with keys from S and values equal to v.
     |      v defaults to None.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
    
    class Graph(delegate.Delegate)
     |  General-purpose graph data structure.
     |  
     |  Method resolution order:
     |      Graph
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, name=None)
     |  
     |  __str__(self)
     |  
     |  add_e(self, e)
     |      Add and return an edge.
     |  
     |  add_v(self, v)
     |      Add and return a vertex.
     |  
     |  all_pairs_sp(self, weight_func=None)
     |      Return a dictionary of shortest path lists for all vertex pairs.
     |      
     |      Keys are (source, destination) tuples.
     |      'weight_func' is a function taking (edge, v1, v2) that returns a weight.
     |      Defaults to e.weight()
     |  
     |  connected_components(self)
     |      Return a list of lists.  Each holds transitively-connected vertices.
     |  
     |  greedy_paths(self, start, goal, weight_func=None)
     |      Return a dict of greedy paths with (start vertex, end vertex) keys.
     |      
     |      Always makes the highest-gain decision.  Will find a path if one exists.
     |      Not necessarily optimal.  'weight_func' is a function of (edge, v1, v2)
     |      returning a weight.  Defaults to e.weight()
     |  
     |  minimal_span_tree(self, **kargs)
     |      Return minimal spanning 'Tree'.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  shortest_tree(self, start, **kargs)
     |      Return a 'Tree' of shortest paths to all nodes.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  breadth_first_search(start_v)
     |      Return a breadth-first search list of vertices.
     |  
     |  depth_first_search(start_v)
     |      Return a depth-first search list of vertices.
     |  
     |  path_weight(path, weight_func=None)
     |      Return the weight of the path, which is a list of vertices.
     |      
     |      'weight_func' is a function taking (edge, v1, v2) and returning a weight.
     |  
     |  topological_sort(start_v)
     |      Return a topological sort list of vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Properties defined here:
     |  
     |  intermed_v
     |      List of all vertices with both incoming and outgoing edges.
     |  
     |      <get> = intermed_v(self)
     |  
     |  sink_v
     |      List of all vertices without outgoing edges.
     |  
     |      <get> = sink_v(self)
     |  
     |  src_v
     |      List of all vertices without incoming edges.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class GraphError(exceptions.StandardError)
     |  Exception for graph errors.
     |  
     |  Method resolution order:
     |      GraphError
     |      exceptions.StandardError
     |      exceptions.Exception
     |  
     |  Methods inherited from exceptions.Exception:
     |  
     |  __getitem__(...)
     |  
     |  __init__(...)
     |  
     |  __str__(...)
    
    class RawEdge(delegate.Delegate)
     |  Base class for undirected and directed edges.  Not directly useful.
     |  
     |  Method resolution order:
     |      RawEdge
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, name, v1, v2)
     |      Name needn't be a string but must be immutable and hashable.
     |  
     |  __setstate__(self, state)
     |      Restore own state and add self to connected vertex edge lists.
     |  
     |  ----------------------------------------------------------------------
     |  Properties defined here:
     |  
     |  all_v
     |      All connected vertices.
     |  
     |      <get> = all_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class SortStruct(__builtin__.float, __builtin__.object)
     |  Throw-away class for temporarily holding values.
     |  
     |  Initialize with a float distance and as many keyword args as desired.
     |  The keyword args become attributes.
     |  
     |  Method resolution order:
     |      SortStruct
     |      __builtin__.float
     |      __builtin__.object
     |  
     |  Static methods defined here:
     |  
     |  __new__(cls, val, **kargs)
     |  
     |  ----------------------------------------------------------------------
     |  Properties defined here:
     |  
     |  number
     |      <get> = number(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__ = <attribute '__weakref__' of 'SortStruct' objects>
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from __builtin__.float:
     |  
     |  __abs__(...)
     |      x.__abs__() <==> abs(x)
     |  
     |  __add__(...)
     |      x.__add__(y) <==> x+y
     |  
     |  __coerce__(...)
     |      x.__coerce__(y) <==> coerce(x, y)
     |  
     |  __div__(...)
     |      x.__div__(y) <==> x/y
     |  
     |  __divmod__(...)
     |      x.__divmod__(y) <==> xdivmod(x, y)y
     |  
     |  __eq__(...)
     |      x.__eq__(y) <==> x==y
     |  
     |  __float__(...)
     |      x.__float__() <==> float(x)
     |  
     |  __floordiv__(...)
     |      x.__floordiv__(y) <==> x//y
     |  
     |  __ge__(...)
     |      x.__ge__(y) <==> x>=y
     |  
     |  __getattribute__(...)
     |      x.__getattribute__('name') <==> x.name
     |  
     |  __getnewargs__(...)
     |  
     |  __gt__(...)
     |      x.__gt__(y) <==> x>y
     |  
     |  __hash__(...)
     |      x.__hash__() <==> hash(x)
     |  
     |  __int__(...)
     |      x.__int__() <==> int(x)
     |  
     |  __le__(...)
     |      x.__le__(y) <==> x<=y
     |  
     |  __long__(...)
     |      x.__long__() <==> long(x)
     |  
     |  __lt__(...)
     |      x.__lt__(y) <==> x<y
     |  
     |  __mod__(...)
     |      x.__mod__(y) <==> x%y
     |  
     |  __mul__(...)
     |      x.__mul__(y) <==> x*y
     |  
     |  __ne__(...)
     |      x.__ne__(y) <==> x!=y
     |  
     |  __neg__(...)
     |      x.__neg__() <==> -x
     |  
     |  __nonzero__(...)
     |      x.__nonzero__() <==> x != 0
     |  
     |  __pos__(...)
     |      x.__pos__() <==> +x
     |  
     |  __pow__(...)
     |      x.__pow__(y[, z]) <==> pow(x, y[, z])
     |  
     |  __radd__(...)
     |      x.__radd__(y) <==> y+x
     |  
     |  __rdiv__(...)
     |      x.__rdiv__(y) <==> y/x
     |  
     |  __rdivmod__(...)
     |      x.__rdivmod__(y) <==> ydivmod(y, x)x
     |  
     |  __repr__(...)
     |      x.__repr__() <==> repr(x)
     |  
     |  __rfloordiv__(...)
     |      x.__rfloordiv__(y) <==> y//x
     |  
     |  __rmod__(...)
     |      x.__rmod__(y) <==> y%x
     |  
     |  __rmul__(...)
     |      x.__rmul__(y) <==> y*x
     |  
     |  __rpow__(...)
     |      y.__rpow__(x[, z]) <==> pow(x, y[, z])
     |  
     |  __rsub__(...)
     |      x.__rsub__(y) <==> y-x
     |  
     |  __rtruediv__(...)
     |      x.__rtruediv__(y) <==> y/x
     |  
     |  __str__(...)
     |      x.__str__() <==> str(x)
     |  
     |  __sub__(...)
     |      x.__sub__(y) <==> x-y
     |  
     |  __truediv__(...)
     |      x.__truediv__(y) <==> x/y
    
    class Tree(Graph)
     |  Tree data structure.
     |  
     |  Edges must be directed and reconvergent paths do not occur.
     |  
     |  Method resolution order:
     |      Tree
     |      Graph
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  add_e(self, e)
     |      Add edge if tree invariant holds.  Otherwise, raise exception.
     |  
     |  auto_add_e(self, e)
     |      Automatically choose correct direction for new edge.
     |  
     |  is_safe_e(self, target_v)
     |      True if edge maintains tree invariant.
     |  
     |  path_dict(self)
     |      Return a dictionary of path lists from the root to each vertex.
     |      
     |      Keys are (source vertex, destination vertex) tuples.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Graph:
     |  
     |  __init__(self, name=None)
     |  
     |  __str__(self)
     |  
     |  add_v(self, v)
     |      Add and return a vertex.
     |  
     |  all_pairs_sp(self, weight_func=None)
     |      Return a dictionary of shortest path lists for all vertex pairs.
     |      
     |      Keys are (source, destination) tuples.
     |      'weight_func' is a function taking (edge, v1, v2) that returns a weight.
     |      Defaults to e.weight()
     |  
     |  connected_components(self)
     |      Return a list of lists.  Each holds transitively-connected vertices.
     |  
     |  greedy_paths(self, start, goal, weight_func=None)
     |      Return a dict of greedy paths with (start vertex, end vertex) keys.
     |      
     |      Always makes the highest-gain decision.  Will find a path if one exists.
     |      Not necessarily optimal.  'weight_func' is a function of (edge, v1, v2)
     |      returning a weight.  Defaults to e.weight()
     |  
     |  minimal_span_tree(self, **kargs)
     |      Return minimal spanning 'Tree'.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  shortest_tree(self, start, **kargs)
     |      Return a 'Tree' of shortest paths to all nodes.
     |      
     |      Keyword arg 'weight_func' is a function taking (edge, v1, v2) and
     |      returning a weight.  Defaults to e.weight().
     |      'targets' list of target vertices.  Defaults to all vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from Graph:
     |  
     |  breadth_first_search(start_v)
     |      Return a breadth-first search list of vertices.
     |  
     |  depth_first_search(start_v)
     |      Return a depth-first search list of vertices.
     |  
     |  path_weight(path, weight_func=None)
     |      Return the weight of the path, which is a list of vertices.
     |      
     |      'weight_func' is a function taking (edge, v1, v2) and returning a weight.
     |  
     |  topological_sort(start_v)
     |      Return a topological sort list of vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from Graph:
     |  
     |  intermed_v
     |      List of all vertices with both incoming and outgoing edges.
     |  
     |      <get> = intermed_v(self)
     |  
     |  sink_v
     |      List of all vertices without outgoing edges.
     |  
     |      <get> = sink_v(self)
     |  
     |  src_v
     |      List of all vertices without incoming edges.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class TreeError(GraphError)
     |  Method resolution order:
     |      TreeError
     |      GraphError
     |      exceptions.StandardError
     |      exceptions.Exception
     |  
     |  Methods inherited from exceptions.Exception:
     |  
     |  __getitem__(...)
     |  
     |  __init__(...)
     |  
     |  __str__(...)
    
    class UndirEdge(RawEdge)
     |  Undirected edge.
     |  
     |  Method resolution order:
     |      UndirEdge
     |      RawEdge
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __str__(self)
     |  
     |  enters(self, v)
     |      True if this edge is connected to the vertex.
     |  
     |  leaves(self, v)
     |      True if this edge is connected to the vertex.
     |  
     |  weight(self, v1, v2)
     |      1 if this edge connects the vertices.
     |  
     |  ----------------------------------------------------------------------
     |  Properties defined here:
     |  
     |  dest_v
     |      Destination vertices.
     |  
     |      <get> = dest_v(self)
     |  
     |  src_v
     |      Source vertices.
     |  
     |      <get> = src_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from RawEdge:
     |  
     |  __init__(self, name, v1, v2)
     |      Name needn't be a string but must be immutable and hashable.
     |  
     |  __setstate__(self, state)
     |      Restore own state and add self to connected vertex edge lists.
     |  
     |  ----------------------------------------------------------------------
     |  Properties inherited from RawEdge:
     |  
     |  all_v
     |      All connected vertices.
     |  
     |      <get> = all_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class Vertex(delegate.Delegate)
     |  General graph vertex.
     |  
     |  All methods colaborate with directed and undirected
     |  edges.  If the edges are undirected, ine == oute.
     |  
     |  Method resolution order:
     |      Vertex
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __getstate__(self)
     |      Need to break cycles to prevent recursion blowup in pickle.
     |      
     |      Dump everything except for edges.
     |  
     |  __init__(self, name)
     |      Name neen't be a string but must be hashable and immutable.
     |  
     |  __str__(self)
     |  
     |  attach_e(self, e)
     |      Attach an edge.
     |  
     |  connecting_e(self, v)
     |      List of edges connecting self and other vertex.
     |  
     |  ----------------------------------------------------------------------
     |  Properties defined here:
     |  
     |  adjacent_v
     |      Set of adjacent vertices.  Edge direction ignored.
     |  
     |      <get> = adjacent_v(self)
     |  
     |  all_e
     |      All edges.
     |  
     |      <get> = all_e(self)
     |  
     |  in_e
     |      Incoming edges.
     |  
     |      <get> = in_e(self)
     |  
     |  in_v
     |      Set of vertices connected by incoming edges.
     |  
     |      <get> = in_v(self)
     |  
     |  is_intermed
     |      True if vertex has incoming and outgoing edges.
     |  
     |      <get> = is_intermed(self)
     |  
     |  is_sink
     |      True if vertex has no outgoing edges.
     |  
     |      <get> = is_sink(self)
     |  
     |  is_src
     |      True if vertex has no incoming edges.
     |  
     |      <get> = is_src(self)
     |  
     |  out_e
     |      Outgoing edges.
     |  
     |      <get> = out_e(self)
     |  
     |  out_v
     |      Set of vertices connected by outgoing edges.
     |  
     |      <get> = out_v(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.
     |  
     |  __weakref__ = <attribute '__weakref__' of 'Delegate' objects>
     |      list of weak references to the object (if defined)
    
    class VertexDict(__builtin__.dict, delegate.Delegate)
     |  Dictionary of vertices.
     |  
     |  Method resolution order:
     |      VertexDict
     |      __builtin__.dict
     |      delegate.Delegate
     |      __builtin__.object
     |  
     |  Methods defined here:
     |  
     |  __delitem__(self, key)
     |      Delete all edges connected to the vertex, along with the vertex.
     |  
     |  __init__(self, graph)
     |  
     |  __setitem__(self, key, val)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __dict__ = <dictproxy object>
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__ = <attribute '__weakref__' of 'VertexDict' objects>
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from __builtin__.dict:
     |  
     |  __cmp__(...)
     |      x.__cmp__(y) <==> cmp(x,y)
     |  
     |  __contains__(...)
     |      D.__contains__(k) -> True if D has a key k, else False
     |  
     |  __eq__(...)
     |      x.__eq__(y) <==> x==y
     |  
     |  __ge__(...)
     |      x.__ge__(y) <==> x>=y
     |  
     |  __getattribute__(...)
     |      x.__getattribute__('name') <==> x.name
     |  
     |  __getitem__(...)
     |      x.__getitem__(y) <==> x[y]
     |  
     |  __gt__(...)
     |      x.__gt__(y) <==> x>y
     |  
     |  __hash__(...)
     |      x.__hash__() <==> hash(x)
     |  
     |  __iter__(...)
     |      x.__iter__() <==> iter(x)
     |  
     |  __le__(...)
     |      x.__le__(y) <==> x<=y
     |  
     |  __len__(...)
     |      x.__len__() <==> len(x)
     |  
     |  __lt__(...)
     |      x.__lt__(y) <==> x<y
     |  
     |  __ne__(...)
     |      x.__ne__(y) <==> x!=y
     |  
     |  __repr__(...)
     |      x.__repr__() <==> repr(x)
     |  
     |  clear(...)
     |      D.clear() -> None.  Remove all items from D.
     |  
     |  copy(...)
     |      D.copy() -> a shallow copy of D
     |  
     |  get(...)
     |      D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
     |  
     |  has_key(...)
     |      D.has_key(k) -> True if D has a key k, else False
     |  
     |  items(...)
     |      D.items() -> list of D's (key, value) pairs, as 2-tuples
     |  
     |  iteritems(...)
     |      D.iteritems() -> an iterator over the (key, value) items of D
     |  
     |  iterkeys(...)
     |      D.iterkeys() -> an iterator over the keys of D
     |  
     |  itervalues(...)
     |      D.itervalues() -> an iterator over the values of D
     |  
     |  keys(...)
     |      D.keys() -> list of D's keys
     |  
     |  pop(...)
     |      D.pop(k[,d]) -> v, remove specified key and return the corresponding value
     |      If key is not found, d is returned if given, otherwise KeyError is raised
     |  
     |  popitem(...)
     |      D.popitem() -> (k, v), remove and return some (key, value) pair as a
     |      2-tuple; but raise KeyError if D is empty
     |  
     |  setdefault(...)
     |      D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D
     |  
     |  update(...)
     |      D.update(E, **F) -> None.  Update D from E and F: for k in E: D[k] = E[k]
     |      (if E has keys else: for (k, v) in E: D[k] = v) then: for k in F: D[k] = F[k]
     |  
     |  values(...)
     |      D.values() -> list of D's values
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from __builtin__.dict:
     |  
     |  __new__ = <built-in method __new__ of type object>
     |      T.__new__(S, ...) -> a new object with type S, a subtype of T
     |  
     |  fromkeys = <built-in method fromkeys of _delegate_meta object>
     |      dict.fromkeys(S[,v]) -> New dict with keys from S and values equal to v.
     |      v defaults to None.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from delegate.Delegate:
     |  
     |  __metaclass__ = <class 'delegate._delegate_meta'>
     |      Sets up delegation private variables.
     |      
     |      Traverses inheritance graph on class construction.  Creates a private
     |      __base variable for each base class.  If delegating to the base class is
     |      inappropriate, uses _no_delegation class.

FUNCTIONS
    DataWrapped(cls)
        Returns a wrapped class with a new 'data' member.

DATA
    PosInf = inf
    __author__ = 'Robert Dick and Kosta Gaitanis'
    __author_email__ = 'dickrp@ece.northwestern.edu and gaitanis@tele.ucl....
    __version__ = '0.4'

VERSION
    0.4

AUTHOR
    Robert Dick and Kosta Gaitanis

    Modified by Boyana Norris (norris@mcs.anl.gov) to work with Python 2.3
