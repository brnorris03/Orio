#!/usr/bin/env python

'''Class to automate delegation decisions based on inheritance graph.

Copyright 2004, Robert Dick (dickrp@ece.northwestern.edu).

Whenever you need to delegate to something, inherit from delegate and use
self.__<base>.<method()> to access the base.  If the delegation was
inappropriate due to reconverging paths in the inheritance graph, the return
value will be None.  In the case of reconverging paths, the left-most call in
the method resolution order will be honored.  The rest will be nulified.  You
can also check to see if the base is the no_delegation object.  Delegate to all
your bases if you need everything in the inheritance graph to be visited.  As
long as one of a class's (transitive) bases inherits from Delegate, that's
enough.

For examples of use, please see the delegate.py file.

Python doesn't yet automate meta-class instantiation.  If you need to inherit
from Delegate and another class that does not have a 'type' metaclass, you'll
need to generate a shared derived metaclass and explicitly use that as your
class's metaclass.  For example:

  import Delegate, qt

  class sip_meta_join(type(Delegate), type(qt.QObject)):
    def __init__(*args):
      type(Delegate).__init__(*args)
      type(qt.QObject).__init__(*args)

  class MyClass(Delegate, qt.QObject):
    __metaclass__ = sip_meta_join
    ...

Please see the license file for legal information.'''


__version__ = '0.1'
__author__ = 'Robert Dick (dickrp@ece.northwestern.edu)'

import os
# By default, Python deprecation warnings are disabled, define the CCA_TOOLS_DEBUG env. variable to enable
try:
    import warnings
    if not ('BOCCA_DEBUG' in list(os.environ.keys()) and os.environ['BOCCA_DEBUG'] == '1'): 
        warnings.filterwarnings("ignore", category=DeprecationWarning)
except:
    pass

def should_call(obj, pos, supr):
    '''Returns bool.  Should 'self' delegate to 'super' at 'pos'?
    
    Determines whether pos is left-most derived of super in MRO.'''

    for c in type(obj).__mro__:
        if supr in c.__bases__:
            return pos is c
    return False

class _no_delegation(object):
    '''All class's attributes are null callable's.'''
    
    _to_base = set(['__bases__', '__name__', '__mro__', '__module__'])

    def __getattribute__(self, attr):
        if attr in _no_delegation._to_base:
            return getattr(object, attr)
        def no_action(*pargs, **kargs): pass
        return no_action

'''Whatever'''
no_delegation = _no_delegation()
'''Whatever'''

class _delegate_meta(type):

    '''Sets up delegation private variables.
    
    Traverses inheritance graph on class construction.  Creates a private
    __base variable for each base class.  If delegating to the base class is
    inappropriate, uses _no_delegation class.'''

    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        visited_supr = set()
        for sub in cls.__mro__[:-1]:
            subnm = sub.__name__.split('.')[-1]
            for supr in sub.__bases__:
                suprnm = supr.__name__.split('.')[-1]
                if supr not in visited_supr:
                    visited_supr.add(supr)
                    deleg = supr
                else:
                    deleg = no_delegation
                setattr(cls, '_%s__%s' % (subnm, suprnm), deleg)


class Delegate(object, metaclass=_delegate_meta):
    '''Inherit from Delegate to get delegation variables on class construction.'''


if __name__ == '__main__':
    class Base(Delegate):
        def __init__(self, basearg):
            self.__Delegate.__init__(self)
            self.basearg = basearg
            print('base')

        def __str__(self): return 'BASE'


    class Left(Base):
        def __init__(self, basearg, leftarg):
            self.__Base.__init__(self, basearg)
            self.leftarg = leftarg
            print('left')

        def __str__(self):
            return ' '.join([_f for _f in (self.__Base.__str__(self), 'LEFT') if _f])


    class Right(Base):
        def __init__(self, basearg):
            self.__Base.__init__(self, basearg)
            print('right')

        def __str__(self):
            return ' '.join([_f for _f in (self.__Base.__str__(self), 'RIGHT') if _f])


    class Der(Left, Right):
        def __init__(self, basearg, leftarg):
            self.__Left.__init__(self, basearg, leftarg)
            self.__Right.__init__(self, basearg)
            print('der')

        def __str__(self):
            return ' '.join([_f for _f in (self.__Left.__str__(self),
                self.__Right.__str__(self), 'DER') if _f])


    print('should print base, left, right, der')
    der = Der('basearg', 'leftarg')

    print('\nshould print base, left')
    left = Left('basearg', 'leftarg')
    
    print('\nshould print base right')
    right = Right('basearg')

    print('\nshould print BASE LEFT RIGHT DER')
    print(der)
