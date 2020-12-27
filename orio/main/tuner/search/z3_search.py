from orio.main.util.globals import *

try:
    import z3
    Globals.have_z3 = True
except Exception as e:
    Globals.have_z3 = False
    raise ImportError( "Could not load z3. Will not be using it." )
#    raise ImportError()  # Silent

class Z3search:

    def __init__( self,  total_dims, axis_names, axis_val_ranges, dim_uplimits, constraints, s ):
        self.axis_names = axis_names
        self.axis_val_ranges = axis_val_ranges
        self.constraints = constraints
        self.total_dims = total_dims
        self.dim_uplimits = dim_uplimits
        self.search = s

        # Initialize the solver itself
        self.variables = []
        self.solver = z3.Solver()
        self.optim = z3.Optimize()
        Globals.z3types = {}
        Globals.z3variables = {} 

        # find what we have on the axis
        self.__addVariableNames( self.axis_names, self.axis_val_ranges )
        self.__addConstraints( self.constraints )

    #-----------------------------------------------------
        
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

    # in case a parameter is not numeric (ie, alphanumeric), we cannot store it
    # as a regular value with constraints in z3. Hence, put the values in a list and
    # use their index in the list instead.
    def __defineNonNumeric( self, name, values ):
        Globals.z3types[name] = list( values )
        return [ i for i in range( len( values ) ) ]
    
    # get the parameter corresponding to a numeric value
    def __numericToNonNumeric( self, name, value, num ):
        return Globals.z3types[name][num]
    
    # get the numeric value corresponding to a parameter
    def __nonNumericToNumeric( self, name, value, param ):
        return Globals.z3types[name].index( param )
    
    # Tests whether a z3 variable is numeric. Returns False for non-numeric ones.
    def z3IsNumeric( self, dimensionNumber ):
        return not self.axis_names[dimensionNumber] in Globals.z3types

    def z3abs( self, x ):
        return z3.If(x >= 0,x,-x)

    #-----------------------------------------------------

    # change constraint from infix notation (used in the input file) to prefix notation (used by z3)
    def __infixToPrefix( self, expr ):
        if "or" in expr or "and" in expr:
            if "and" in expr:
                operandsAND = expr.split( "and" )
                return "z3.And(" + ", ".join( [self. infixToPrefix( op ) if "or" in op or "and" in op else op  for op in operandsAND  ] ) + " ) "
            elif "or" in expr:
                operandsOR = expr.split( "or" )
                return  "z3.Or(" + ", ".join( [self. infixToPrefix( op ) if "or" in op or "and" in op else op for op in operandsOR  ] ) + " ) "
            else:
                return expr
        else:
            return expr
        
    #-----------------------------------------------------

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

    # Add a forbidden point in the solver (and the optimizer)
    # Input: a dictionary of performance parameters.
    def addPoint( self, point ):
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
        return

    #-----------------------------------------------------

    # Returns a *feasible* random point
    def getRandomCoord_z3( self ):
        if self.solver.check() == z3.unsat:
            return None
        else:
            model = self.solver.model()
            return model

    # Getting a random point from an SMT solver is not a trivial question.
    # This one gives a somehow random point, but the distribution is not uniform.
    # TODO: David's idea involving SAT.
    # TODO: non-constrained variables?
    def getRandomCoord_z3_distance( self ):
        if self.solver.check() == z3.unsat:
            return None
        else:
            # Pick a random point in the solution space
            rpc = []
            for i in range( self.total_dims ):
                iuplimit = self.dim_uplimits[i]
                ipoint = self.getRandomInt(0, iuplimit-1)
                rpc.append(ipoint)

            # Get the nearest feasible point
            point = self.getNearestFeasible( rpc )
            return self.perfParamTabToCoord( point )

    # This is copied from search.py. Not exactly the cleanest way to do it, but simpler for the moment.
    def getRandomInt(self, lbound, ubound):
        '''To generate a random integer N such that lbound <= N <= ubound'''
        from random import randint

        if lbound > ubound:
            err('orio.main.tuner.search.search internal error: the lower bound of genRandomInt must not be ' +
                   'greater than the upper bound')
        return randint(lbound, ubound)
    
    #-----------------------------------------------------

    # For a given point, returns the nearest feasible point
    def getNearestFeasible( self, coord ):
        # get the variables corresponding to the axis names
        for i in self.axis_names:
            locals()[i] = Globals.z3variables[i]

        # Convert into parameters
        rpp = self.search.coordToPerfParams( coord )
        for i, name in enumerate( self.axis_names ):
            if not self.z3IsNumeric( i ):
                rpp[name] = Globals.z3types[name].index( rpp[name] )
                    
        # create a new scope
        self.optim.push()

        # Get a possible point that minimizes the 1-norm distance to this random point
        # forget the booleans
        self.optim.minimize( z3.Sum( [ self.z3abs( locals()[name] - rpp[name] )for name in self.axis_names  if not z3.is_bool( locals()[name] ) ] ) )
        if z3.unsat == self.optim.check():
            return None
        model = self.optim.model()

        # restore the state (exit the scope)
        self.optim.pop()
        return self.z3ToPoint( model )

    #-----------------------------------------------------
        
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
        
    #-----------------------------------------------------

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

    def perfParamToCoord( self, params ):
        coord = [None]*self.total_dims
        for i in range( self.total_dims ):
            coord[i] = params[self.axis_names[i]]
            if not self.z3IsNumeric( i ):
                name = self.axis_names[i]
                coord[i] = Globals.z3types[name].index( coord[i] )
        return coord

    def coordToTabOfPerfParams( self, coord ):
        params = self.search.coordToPerfParams( coord )
        perftab = []
        for i, name in enumerate( self.axis_names ):
            c = params[name]
            if not self.z3IsNumeric( i ):
                c = Globals.z3types[name].index( c )
            perftab.append( c )
        return perftab

    def perfParamTabToCoord( self, params ):
        coord = [None]*self.total_dims
        for i in range( self.total_dims ):
            coord[i] = self.axis_val_ranges[i].index( params[i] )
        return coord


