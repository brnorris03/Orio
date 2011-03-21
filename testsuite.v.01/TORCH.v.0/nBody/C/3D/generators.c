
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "nBodyUtil.h"
#include "generators.h"

/*
 Input generators for particle methods.

 Alex Kaiser, LBNL, 10/2010
 */



double set_size( int n ){
	/*
	 Set global parameter size according to input.
	 Keep density constant.

	 Input:
	 int n                      Number of particles

	 Output:
	 double size (returned)     Global parameter for boundaries of region.
	 */
    size = sqrt( density * n );
    return size;
}



void init_particles( int n, particle_t *p ){
	/*
	 Initialize the particle positions and velocities.
	 Current implementation resets random number generator bcnrand for reproducible results.

	 Input:
	 int n              Number of particles.
	 particle_t *p      Array of particles, must be preallocated.

	 Parameters:
	 double velocityCoefficient    Velocities are set uniformly in the interval:
	                                   ( -velocityCoefficient, velocityCoefficient )

	 Output:
	 Particles are set to randomized initial positions and velocities.
	 */

	double velocityCoefficient = 1.0; // scale velocities by this amount

	// reset random number generator before use
	resetBcnRand() ;

	int i;
    for(i=0; i<n; i++){
		// assign with known random values for verification
    	p[i].x = size * bcnrand() ;
    	p[i].y = size * bcnrand() ;
    	p[i].z = size * bcnrand() ;

    	p[i].vx = velocityCoefficient * ( 2.0 * bcnrand() - 1.0) ;
    	p[i].vy = velocityCoefficient * ( 2.0 * bcnrand() - 1.0) ;
    	p[i].vz = velocityCoefficient * ( 2.0 * bcnrand() - 1.0) ;

		p[i].mass = MASSDEF;
    }

}


