import orio.main.tuner.search.search
from orio.main.util.globals import *
import time, types

import itertools
import math

class Direct( orio.main.tuner.search.search.Search ):

    def __init__( self, params ):
        orio.main.tuner.search.search.Search.__init__(self, params)
        # rate-of-change
        self.K_roc = .5
        # Difference between the current minimum and the "guessed" absolute minimum
        # such that f* <= fmin - epsilon fmin
        self.epsilon = 1e-4


    def searchBestCoord( self, startCoord = None ):

        # We are in a hyperrectangle. Initialization: take the whole parameter space.

        rectangle = [ [ 0, self.dim_uplimits[i] ] for i in range( self.total_dims ) ]
        print "initial rectangle", rectangle
        fmin = float( 'inf' )
        rectangles = [ rectangle ]
        minpoint = self.dim_uplimits

        start_time = time.time()
        runs = 0

        # Keep the rectangles that are of the same measure
        # key: measure (half longuest diagonal length)
        # value: list of tuples ( rectangle, value at the center ) )
        rect_sizes = {}

        # initialize
        
        center = self.__getCentroid( rectangle )
        cost = self.getPerfCost( center )
        fc = sum( cost ) / len( cost )
        dist = 0
        for c in rectangle:
            dist = max( dist, self.__distance( c, center ) )
        rect_sizes[ dist ] = [ ( rectangle, fc ) ]
            
        
        while True:
            if rectangles == []:
                break
            rectangle = rectangles.pop( 0 )
            runs += 1
        
            # Trisect the rectangle along the longuest dimension

            longuest_len, longuest_dim = max( ( x, i ) for i,x in enumerate( [ i[1] - i[0] for i in rectangle ] ) )

            if 0 == int( round( longuest_len / 3 ) ):
                break
        
            rec1 = rectangle[:]
            rec1[longuest_dim] = rectangle[longuest_dim][:]
            rec1[longuest_dim][1] = rectangle[longuest_dim][0] + int( round( longuest_len / 3 ) ) # DIRTY
            corners = list( itertools.product( *rec1, repeat=1 ))
            cor1 = [ list( c ) for c in corners ]
            r1 = ( rec1, cor1 )

            rec2 = rectangle[:]
            rec2[longuest_dim] = rectangle[longuest_dim][:]
            rec2[longuest_dim][0] = rectangle[longuest_dim][0] + int( round( longuest_len / 3 ) )
            rec2[longuest_dim][1] = rectangle[longuest_dim][0] + int( round( 2 * longuest_len / 3 ) )
            corners = list( itertools.product( *rec2, repeat=1 ))
            cor2 = [ list( c ) for c in corners ]
            r2 = ( rec2, cor2 )

            rec3 = rectangle[:]
            rec3[longuest_dim] = rectangle[longuest_dim][:]
            rec3[longuest_dim][0] = rectangle[longuest_dim][0] + int( round( 2 * longuest_len / 3 ) )
            corners = list( itertools.product( *rec3, repeat=1 ))
            cor3 = [ list( c ) for c in corners ]
            r3 = ( rec3, cor3 )

            print "Dividing rectangle", rectangle, "into", rec1, "AND", rec2, "AND", rec3
            print "With corners", cor1, "AND", cor2, "AND", cor3

            # Select the potentially optimal rectangles

            new_fmin = fmin
            fstar = ( 1 - self.epsilon ) * fmin
            for rec, cor in r1, r2, r3:
                

                print "working in rectangle: ", rec, "corners", cor

                # Take the center
                center = self.__getCentroid( cor )

                # Evaluate the perf at the center
                cost = self.getPerfCost( center )
                fc = sum( cost ) / len( cost )
                dist = 0
                for c in cor:
                    dist = max( dist, self.__distance( c, center ) )
                print "fc", fc, "dist", dist

                # Add it to the dictionnary
                if rect_sizes.has_key( dist ):
                    rect_sizes[ dist ].append( ( cor, fc ) )
                else:
                    rect_sizes[ dist ] = [ ( cor, fc ) ]
                    
                s = sorted( rect_sizes.keys() )
                if rect_sizes.has_key( dist ):
                    i = s.index( dist )
                else:
                    for i in s:
                        if i > dist:
                            break
                    
                # rectangles smaller than the current one
                I1 = { k:v for k,v in rect_sizes.items() if k in s[:i]}
                # rectangles larger than the current one
                if i < len( rect_sizes.keys() ):
                    I2 = { k:v for k,v in rect_sizes.items() if k in s[i+1:]}
                else:
                    I2 = {}
                # rectangles as big as than the current one
                if rect_sizes.has_key( dist ):
                    I3 = rect_sizes[ dist ]
                else:
                    I3 = []

                opt = True

                # equation (6)
                
                if I3 != []:
                    for i in I3:
                        if i[1] < fc:
                            opt = False

                if opt == False:
                    # Not optimal
                    continue

                # equation (7)
                
                maxI1 = 0
                for i in I1:
                    for r in I1[i]:
                        value = abs( ( r[1] - fc ) / ( i - dist ) )
                        if value > maxI1:
                            maxI1 = value

                minI2 = float( 'inf' )
                for i in I2:
                    for r in I2[i]:
                        value = abs( ( r[1] - fc ) / ( i - dist ) )
                        if value < minI2:
                            minI2 = value

                if maxI1 > minI2:
                    opt = False
                    continue

                # equation (8)

                if fmin != 0:
                    value = ( fmin - fc ) + dist * minI2 
                    value /= abs( fmin )
                    if value < self.epsilon:
                        opt = False
                        continue
                # equation (9)
                else:
                    if fc > dist * minI1:
                        opt = False
                        continue

                # If we are still here, the conditions are fulfilled. The rectangle is potentially optimal.
                # Add it (it will be divided later).
                print "potentially optimal rectangle found", rec
                rectangles.append( rec )

                # do we have the minimum?
                if I1 == {}:
                    if fc < new_fmin:
                        new_fmin = fc
                        minpoint = center
                
            # Remove the big rectangle from the dictionnary
            for r in rect_sizes[ dist ]:
                if r[0] == rectangle:
                    rect_sizes[ dist ].remove( r )
                    break
                
            fmin = new_fmin
                
        search_time = time.time() - start_time

        return minpoint, fmin, search_time, runs





    
    def __distance( self, p1, p2 ):
        d = 0
        for c1, c2 in zip( p1, p2 ):
            d += ( c1 - c2 ) * ( c1 - c2 )
        d = math.sqrt( d )
        return d

    def __getCentroid(self, coords):
        '''Return a centroid coordinate'''
        if self.have_z3:
            model = self.getCentroidZ3( coords )
            point = self.z3ToPoint( model )
            return self.perfParamToCoord( point )
        
        total_coords = len(coords)
        centroid = coords[0]
        for c in coords[1:]:
            centroid = self.addCoords(centroid, c)
        centroid = self.mulCoords((1.0/total_coords), centroid)
        return centroid

