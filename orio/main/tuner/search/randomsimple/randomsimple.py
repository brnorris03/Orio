# Very basic random search

import sys, time
import math
import random
import orio.main.tuner.search.search
from orio.main.util.globals import *

try:
    import z3
    _have_z3 = True
except Exception as e:
    _have_z3 = False
    
class Randomsimple(orio.main.tuner.search.search.Search):
    def __init__(self, params):

        random.seed(1)
        orio.main.tuner.search.search.Search.__init__(self, params)

        self.__readAlgoArgs()
        if _have_z3:
            self.have_z3 = True
            self.initZ3()
        else:
            self.have_z3 = False
            self.solver = None
            self.optim = None

        if self.time_limit <= 0 and self.total_runs <= 0:
            err(('orio.main.tuner.search.randomsimple.randomsimple: %s search requires either (both) the search time limit or (and) the ' +
                    'total number of search runs to be defined') % self.__class__.__name__)

    def searchBestCoord(self, startCoord=None):
        
        info('\n----- begin random search -----')
        print( "Total runs:", self.total_runs )
        print( "Time limit:", self.time_limit )
        
        bestperfcost = self.MAXFLOAT
        bestcoord = None
        runs = 0

        # start the timer
        start_time = time.time()
        init = True

        visited = []
        
        while ( self.time_limit < 0 or ( time.time() - start_time ) < self.time_limit ) and ( self.total_runs < 0 or runs < self.total_runs ):
            # get a random point
            coord = self.getRandomCoord()
            
            if not self.have_z3 and not self.checkValidity( coord ) or coord in visited:
                print( "invalid point" )
                continue
            try:
                print( "coord:", coord, "run", runs )
                perf_costs = self.getPerfCost( coord )
                if bestperfcost > sum( perf_costs ):
                    info( "Point %s gives a better perf: %s -- %s" % (coord, sum( perf_costs ), bestperfcost ) )
                    bestperfcost = sum( perf_costs )
                    bestcoord = coord
            except Exception, e:
                info('FAILED: %s %s' % (e.__class__.__name__, e))
            runs += 1
            if not self.have_z3:
                visited.append( coord )
            else:
                point = self.coordToPerfParams( coord )
                self.__addPoint( point )

        search_time = time.time() - start_time
        return bestcoord, bestperfcost, search_time, runs

    def checkValidity( self, coord ):
        perf_params = self.coordToPerfParams(coord)
        try:
            is_valid = eval(self.constraint, perf_params, dict(self.input_params))
        except Exception, e:
            err('failed to evaluate the constraint expression: "%s"\n%s %s' % (self.constraint,e.__class__.__name__, e))
            return False
        return is_valid
    
    def __readAlgoArgs(self):
        for vname, rhs in self.search_opts.iteritems():
            if vname == 'total_runs':
                self.total_runs = rhs
            else:
                err('orio.main.tuner.search.randomsimple: unrecognized %s algorithm-specific argument: "%s"' %
                    (self.__class__.__name__, vname))

    def initZ3(self):
        # Initialize the solver itself
        self.variables = []
        self.solver = z3.Solver()
        self.optim = z3.Optimize()
        Globals.z3types = {}
        Globals.z3variables = {} 

        # find what we have on the axis
        self.__addVariableNames( self.axis_names, self.axis_val_ranges )
        self.__addConstraints( self.params['ptdriver'].tinfo.pparam_constraints )
                
    # In this function, either z3 finds a feasible point, or we ask the default function draw one.
    def __getRandomPoint(self):
        if not self.have_z3:
            self.getRandomCoord()
        else:
            # if I have z3, get a *feasible* random coord
            model = self._getRandomCoordZ3()
            #model = self._getRandomCoordZ3Distance()
            point = self._z3ToPoint( model )
            coord = self._perfParamTabToCoord( point )
            # If I could not find any feasible point, just return a random point
            if None != coord:
                return coord
            
    # Defines the variables, their type and, for numeric ones, their definition domain
    def __addVariableNames( self, names, ranges ):
        for idx, name in enumerate( names ):
            values = ranges[idx]
            if ( [ True, False ] == values ) or ( [ False, True ] == values ): #len( values ) == 2 and ( False in values or True in values ):
                Globals.z3variables[name] = z3.Bool( name )
                self.__addDefinitionDomain( name, values )
            else:
                try:
                    toto = int( values[0] )
                    Globals.z3variables[name] = z3.Int( name )
                    self.__addDefinitionDomain( name, values )
                except ValueError:
                    # if the parameter is non-numeric
                    # TODO we can add a case to handle floats
                    Globals.z3variables[name] = z3.Int( name )
                    numvalues = self.__defineNonNumeric( name, values )
                    self.__addDefinitionDomain( name, numvalues )
            self.variables.append( name ) # FIXME variables is axis_names
        return
    
    # z3-specific routine:
    # Defines a variable's definition domain
    def __addDefinitionDomain( self, var, dom ):
        # get the variables
        for k,v in Globals.z3variables.iteritems( ):
            locals()[k] = v
        if len( dom ) == 0:
            return
        definition = z3.Or( [ eval( "(" + var + " == " + str( v ) + ")" ) for v in dom ] )
        # definition = z3.And( [ v >= 0, v < len( dom ) ] )
        
        self.solver.add( definition )
        self.optim.add( definition )
        return
        
    # z3-specific routine:
    # Add the constraints
    def __addConstraints( self, constraints ):
        # get the variables
        for k,v in Globals.z3variables.iteritems( ):
            locals()[k] = v
        for vname, rhs in constraints:
            # convert to constraint to prefix syntax
            toto = eval( self.__infixToPrefix( rhs ) )
            self.solver.add( toto )
            self.optim.add( toto )
        return

    # z3-specific routine:
    # in case a parameter is not numeric (ie, alphanumeric), we cannot store it
    # as a regular value with constraints in z3. Hence, put the values in a list and
    # use their index in the list instead.
    def __defineNonNumeric( self, name, values ):
        Globals.z3types[name] = list( values )
        return [ i for i in range( len( values ) ) ]
    
    # z3-specific routine:
    # get the parameter corresponding to a numeric value
    def __numericToNonNumeric( self, name, value, num ):
        return Globals.z3types[name][num]
    
    # z3-specific routine:
    # get the numeric value corresponding to a parameter
    def __nonNumericToNumeric( self, name, value, param ):
        return Globals.z3types[name].index( param )
    
    # z3-specific routine:
    # change constraint from infix notation (used in the input file) to prefix notation (used by z3)
    def __infixToPrefix( self, expr ):
        if "or" in expr or "and" in expr:
            if "and" in expr:
                operandsAND = expr.split( "and" )
                return "z3.And(" + ", ".join( [ infixToPrefix( op ) if "or" in op or "and" in op else op  for op in operandsAND  ] ) + " ) "
            elif "or" in expr:
                operandsOR = expr.split( "or" )
                return  "z3.Or(" + ", ".join( [ infixToPrefix( op ) if "or" in op or "and" in op else op for op in operandsOR  ] ) + " ) "
            else:
                return expr
        else:
            return expr
        
    # Add a forbidden point in the solver (and the optimizer)
    # Input: a dictionary of performance parameters.
    def __addPoint( self, point ):
        # get the variables corresponding to the axis names
        for name in self.axis_names:
            locals()[name] = Globals.z3variables[name]
        point2 = {}
        # if there is a non-numeric value here, translate it
        for idx, name in enumerate( point.keys() ):
            # translate non-numeric values
            if not self.z3IsNumeric( idx ): # non-numeric points -> coord to value
                elem = Globals.z3types[name].index( point[name] )
            else:
                elem = point[name]
            point2[name] = elem
            
        # remove this point
        for s in [ self.solver, self.optim ]:
            s.add( z3.Or( [ locals()[name] != point2[name] for name in self.axis_names ] ) )

    # Tests whether a z3 variable is numeric. Returns False for non-numeric ones.
    def z3IsNumeric( self, dimensionNumber ):
        return not self.axis_names[dimensionNumber] in Globals.z3types

    # Returns a *feasible* random point
    def _getRandomCoordZ3( self ):
        if self.solver.check() == z3.unsat:
            return None
        else:
            model = self.solver.model()
            return model

    # Converts a model (returned by z3) into a point.
    # Returns a list
    def z3ToPoint( self, point ):
        res = []
        # get the variables corresponding to the axis names
        for idx, i in enumerate( self.axis_names ):
            locals()[i] = Globals.z3variables[i]
            if i in Globals.z3types: # non-numeric points -> coord to value
                value = Globals.z3types[i][point[locals()[i]].as_long()]
            elif None == point[locals()[i]]:
                info( "no value for %s, take %r (incomplete model)" % ( self.axis_names[idx], self.axis_val_ranges[idx][0] ) )
                value = self.axis_val_ranges[idx][ 0]
            elif z3.is_int_value( point[locals()[i]] ):
                value = point[locals()[i]].as_long()
            elif z3.is_true( point[locals()[i]] ) or z3.is_false( point[locals()[i]] ):
                value = z3.is_true( point[locals()[i]] )
            res.append( value )
        return res

    def perfParamTabToCoord( self, params ):
        coord = [None]*self.total_dims
        for i in range( self.total_dims ):
            coord[i] = self.axis_val_ranges[i].index( params[i] )
        return coord


