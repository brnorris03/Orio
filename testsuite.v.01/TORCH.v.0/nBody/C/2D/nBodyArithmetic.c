
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nBodyUtil.h"
#include "generators.h"
#include "nBodyArithmetic.h"

/*
 Arithmetic and run routines for N Body calculations.

 Alex Kaiser, LBNL, 10/2010.
 */



void runNaive(int n, int numSteps, int forceType, char output, int saveFreq, char verify, forceLog currentLog){
	/*
	 Runs simulation of N-Body problem using naive algorithm.

	 Input:
	     int n                  Number of particles to simulate.
	     int numSteps           Number of steps to simulate.
	     int forceType          Which force to use.
	     char output            If true, this routine will output a text file of the positions of the particles.
	     int saveFreq           if (output) data is written every this many time steps.
	     char verify            If true, this routine store forces for verification purposes
	     forceLog currentLog    Preallocated space of size (2 x numToCheck) for output of forces.
	                                if (!verify) this may be passed null.

	 Output:
	     if( output ), a text file "naive.txt" is written with the positions of the particles at each step.
	     if( verify ), the arrays in the structure forceLog will be filled with force data for comparisons
	 */

	printf("Beginning simulation with naive algorithm.\n");
	if(output)
		printf("Output is on.\n") ;
	else
		printf("Output is off.\n") ;
	if(verify)
		printf("Force log for verification is on.\nLogging first %d particles and %d timesteps.\n",
				                                        currentLog.length, currentLog.steps) ;
	else
		printf("Force log for verification is off.\n") ;


	FILE *fsave = fopen( "naive.txt", "w" );

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );



    int step, i, j;

    // if force is not a bounce type, adjust the walls of the region.
    if(forceType != 1){
    	// push particles to the middle of the region
		size *= 5.0; // display a big window around where the particles actually are
		for (i = 0; i < n; i++) {
			particles[i].x += size / 2.0;
			particles[i].y += size / 2.0;
		}
    }

    //  simulate a number of time steps
    double simulation_time = read_timer( );

	// store initial conditions to file
    if(output)
    	save( fsave, n, particles );

    for(step = 0; step < numSteps; step++ ){

        //  compute forces
        for(i = 0; i < n; i++ ){
            particles[i].ax = 0.0;
			particles[i].ay = 0.0;
            for (j = 0; j < n; j++ ){
				if( i != j){
					apply_force( particles+i, particles+j, forceType );  // reference the appropriate particles using pointer arithmetic
				}
			}
        }

        // log forces if requested
        if( verify && (step < currentLog.steps) ){
        	for(i=0; i<currentLog.length; i++){
        		currentLog.xForces[step][i] = particles[i].ax * particles[i].mass ;
        		currentLog.yForces[step][i] = particles[i].ay * particles[i].mass ;
        	}
        }

        //  move particles
        for(i = 0; i < n; i++ )
            move( particles+i, forceType);

        //  save if necessary
        if( output && fsave && ((step % saveFreq) == 0) )
            save( fsave, n, particles );

		if (step % 100 == 0)
        	printf("step = %d\n", step);
    }


    simulation_time = read_timer( ) - simulation_time;

    printf( "n = %d, simulation time = %g seconds.\n\n", n, simulation_time );

    free( particles );
    if( fsave )
        fclose( fsave );

}


void runCutoff(int n, int numSteps, int forceType, char output, int saveFreq, char verify, forceLog currentLog){
	/*
	 Runs simulation of N-Body problem using naive algorithm.

	 Input:
	     int n                  Number of particles to simulate.
	     int numSteps           Number of steps to simulate.
	     int forceType          Which force to use.
	     char output            If true, this routine will output a text file of the positions of the particles.
	     int saveFreq           if (output) data is written every this many time steps.
	     char verify            If true, this routine store forces for verification purposes
	     forceLog currentLog    Preallocated space of size (2 x numToCheck) for output of forces.
	                                if (!verify) this may be passed null.

	 Output:
	     if( output ), a text file "cutoff.txt" is written with the positions of the particles at each step.
	     if( verify ), the arrays in the structure forceLog will be filled with force data for comparisons
	 */

	printf("Beginning simulation with cutoff algorithm.\n");
	if(output)
		printf("Output is on.\n") ;
	else
		printf("Output is off.\n") ;
	if(verify)
		printf("Force log for verification is on.\nLogging first %d particles and %d timesteps.\n",
				                                        currentLog.length, currentLog.steps) ;
	else
		printf("Force log for verification is off.\n") ;


	FILE *fsave = fopen( "cutoff.txt", "w" );

	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

	bin *bins;
	int numBins;
	bins = initBins(&numBins);

    int step, i, j, k;
	int localIndex;
	int neighborIndex;

	// if force is not a bounce type, adjust the walls of the region.
    if(forceType != 1){
    	// push particles to the middle of the region
		size *= 5.0; // display a big window around where the particles actually are
		for (i = 0; i < n; i++) {
			particles[i].x += size / 2.0;
			particles[i].y += size / 2.0;
		}
    }

    //  simulate a number of time steps
    double simulation_time = read_timer( );

	// store initial conditions to file
	save( fsave, n, particles );

    for(step = 0; step < numSteps; step++ ){

		bins = assignBins(particles, bins, n);

		for(i=0; i<n; i++){
			particles[i].ax = 0.0;
			particles[i].ay = 0.0;
		}

		for(k=0; k<numBins; k++){

			//compare to local particles
			for(i=0; i<bins[k].numParticles; i++){
				for(j=0; j<bins[k].numParticles; j++){
					localIndex = bins[k].parts[i] ;
					neighborIndex = bins[k].parts[j] ;
					if( localIndex != neighborIndex ){
						apply_force( particles+localIndex, particles+neighborIndex, forceType );
					}
				}
			}

			// compare to left bin
			if( k > 0 ){
				for(i=0; i<bins[k].numParticles; i++){
					for(j=0; j<bins[k-1].numParticles; j++){
						localIndex = bins[k].parts[i] ;
						neighborIndex = bins[k-1].parts[j] ;
						if(localIndex != neighborIndex){
							apply_force(particles+localIndex, particles+neighborIndex, forceType);
						}
					}
				}
			}


			// compare to right bin
			if( k < numBins-1 ){
				for(i=0; i<bins[k].numParticles; i++){
					for(j=0; j<bins[k+1].numParticles; j++){
						localIndex = bins[k].parts[i] ;
						neighborIndex = bins[k+1].parts[j] ;
						if(localIndex != neighborIndex){
							apply_force(particles+localIndex, particles+neighborIndex, forceType);
						}
					}
				}
			}

		}

        // log forces if requested
        if( verify && (step < currentLog.steps) ){
        	for(i=0; i<currentLog.length; i++){
        		currentLog.xForces[step][i] = particles[i].ax * particles[i].mass ;
        		currentLog.yForces[step][i] = particles[i].ay * particles[i].mass ;
        	}
        }

        //  move particles
        for(i = 0; i < n; i++ )
            move( particles+i , forceType);

        //  save if necessary
        if( output && fsave && ((step % saveFreq) == 0) )
            save( fsave, n, particles );

		if (step % 100 == 0)
        	printf("step = %d\n", step);

		// clean bins for next iteration
		bins = cleanBins(bins, numBins);
    }


    simulation_time = read_timer( ) - simulation_time;

    printf( "n = %d, simulation time = %g seconds.\n\n", n, simulation_time );

    free( particles ) ;
    free( bins ) ;
    if( fsave )
        fclose( fsave );

}


void runBarnesHut(int n, int numSteps, int forceType, char output, int saveFreq, char verify, forceLog currentLog){
	/*
	 Runs simulation of N-Body problem using the Barnes Hut algorithm.

	 Input:
	     int n                  Number of particles to simulate.
	     int numSteps           Number of steps to simulate.
	     int forceType          Which force to use.
	     char output            If true, this routine will output a text file of the positions of the particles.
	     int saveFreq           if (output) data is written every this many time steps.
	     char verify            If true, this routine store forces for verification purposes
	     forceLog currentLog    Preallocated space of size (2 x numToCheck) for output of forces.
	                                if (!verify) this may be passed null.

	 Output:
	     if( output ), a text file "barnesHut.txt" is written with the positions of the particles at each step.
	     if( verify ), the arrays in the structure forceLog will be filled with force data for comparisons
	 */

	printf("Beginning simulation with Barnes Hut algorithm.\n");
	if(output)
		printf("Output is on.\n") ;
	else
		printf("Output is off.\n") ;
	if(verify)
		printf("Force log for verification is on.\nLogging first %d particles and %d timesteps.\n",
				                                        currentLog.length, currentLog.steps) ;
	else
		printf("Force log for verification is off.\n") ;


	FILE *fsave = fopen( "barnesHut.txt", "w" );

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );



    int step, i;

    // if force is not a bounce type, adjust the walls of the region.
    if(forceType != 1){
    	size *= 5.0; // display a big window around where the particles actually are
    	// push particles to the middle of the region
		for (i = 0; i < n; i++) {
			particles[i].x += size / 2.0;
			particles[i].y += size / 2.0;
		}
    }


    node * quadTree;

    //  simulate a number of time steps
	double simulation_time = read_timer();

	// store initial conditions to file
	save(fsave, n, particles);

	for (step = 0; step < numSteps; step++) {

		//  compute forces
		quadTree = quadTreeBuild(particles, n);
		*quadTree = computeCMandTM(*quadTree, particles);

		// recursively compute forces
		for (i = 0; i < n; i++) {
			particles[i].ax = 0.0;
			particles[i].ay = 0.0;
			treeForce(particles, i, *quadTree, forceType);
		}

        // log forces if requested
        if( verify && (step < currentLog.steps) ){
        	for(i=0; i<currentLog.length; i++){
        		currentLog.xForces[step][i] = particles[i].ax * particles[i].mass ;
        		currentLog.yForces[step][i] = particles[i].ay * particles[i].mass ;
        	}
        }

		//  move particles
		for (i = 0; i < n; i++)
			move(particles + i, forceType);

        //  save if necessary
        if( output && fsave && ((step % saveFreq) == 0) )
            save( fsave, n, particles );

		if (step % 100 == 0)
			printf("step = %d\n", step);

		freeDescendents(*quadTree);
	}

	simulation_time = read_timer() - simulation_time;

	printf("theta = %f\n", THETA);
	printf( "n = %d, simulation time = %g seconds.\n\n", n, simulation_time );

	free(particles);
	if (fsave)
		fclose(fsave);


}


void apply_force( particle_t *particle, particle_t *neighbor, int type){
	/*
	 Applies force to a particle according to a neighboring particle.
	 Changing the force drastically changes results.

	 Input:
	 particle_t *particle      Current particle.
	 particle_t *neighbor      Its neighbor to interacte with.
	 int type                  Parameter to control type of force.
	 	 	                       Type 0 = gravity-like attractive force.
	 	 	                       Optionally, this force cuts off at a small radius to keep forces from exploding at short range.
	 	 	                       This is default behavior.
	                               Type 1 = repulsive.

	 Output:
	 Acceleration of current particle is updated.
	 */

	if(type == 0 || type > 1){
		// Default force.
		// simple gravity type attractive force
		double dx = neighbor->x - particle->x;
		double dy = neighbor->y - particle->y;

		double G = 9.8;
		double distance = sqrt(dx * dx + dy * dy);
		double force;

		if (distance <= 0.001) {
			force = (G * particle->mass * neighbor->mass) / (0.001 * 0.001);
			particle->ax += dx * force;
			particle->ay += dy * force;

		} else {
			force = (G * particle->mass * neighbor->mass) / (distance * distance);
			particle->ax += dx * force;
			particle->ay += dy * force;
		}
	}

	else if(type == 1){
		// simple, short-range repulsive force
		double dx = neighbor->x - particle->x;
		double dy = neighbor->y - particle->y;
		double r2 = dx * dx + dy * dy;

		if (r2 > cutoff * cutoff)
			return;

		r2 = fmax(r2, min_r * min_r );
		double r = sqrt(r2);

		//  very simple short-range repulsive force
		double coef = (1 - cutoff / r) / r2 / particle->mass;

		// note: this force is equal to:
		// r - cutoff / (r * ||r||^2)
		// 1/||R||^2 - cutoff / r^3

		particle->ax += coef * dx;
		particle->ay += coef * dy;
	}

}


void treeForce( particle_t *particles, int j, node nn, int forceType ){
	/*
	Compute force on particle j due to all particles inside node n
	Must initialize the acceleration to zero on each particle before calling

	Input:
    particle_t *particles       Array of particles.
    int j                       Index of particle of which to compute acceleration.
    node nn                     Root of subtree to interact particle with.
    int forceType               Force type to use for non-recursive call.
                                    Must be set to use gravity type force.

    Output:
    Acceleration of current particle is updated.
	 */

	// make sure force type matches internal calculations.
	if( forceType == 1 ){
		fprintf(stderr, "Must use gravity type force for current implementation of Barnes Hut, since force is hard coded in tree force.\nExiting.\n\n") ;
		exit(-1) ;
	}

	if(nn.particleIndex != -1){ // if node has a particle, interact with it
		if( j != nn.particleIndex){ // don't interact particle with itself
			apply_force(particles+j, particles + nn.particleIndex, forceType) ;
		}
	}
	else if (nn.isInternal){

		if( nn.totalMass == 0.0){
			printf("error = total mass of internal node is 0\n");
		}

		// check distances to center of mass to see if node is adequately far
		double dx = nn.centerMass[0] - particles[j].x ;
		double dy = nn.centerMass[1] - particles[j].y ;
		double distance = sqrt(dx*dx + dy*dy);

		if( ((2.0*nn.radius) / distance) < THETA ){
			// node is far enough to be treated as a particle

			// allow a force cutoff, though it shouldn't be called often
			// this prevents huge forces when particles are close together
			// not physical, must modify for any physical modeling
			char cutoffAllowed = 1;

			if( cutoffAllowed ){
				if (distance <= 0.001){
					//printf("this actually passed on recursion.\n");
					double force = 9.8 * particles[j].mass * nn.totalMass / (0.001*0.001) ;
					particles[j].ax += dx * force;
					particles[j].ay += dy * force;

				}
				else{
					double force = 9.8 * particles[j].mass * nn.totalMass / (distance*distance) ;
					particles[j].ax += dx * force;
					particles[j].ay += dy * force;
				}
			}

			// no cutoff allowed, std gravitional force.
		    else{
				double force = 9.8 * particles[j].mass * nn.totalMass / (distance*distance) ;
				particles[j].ax += dx * force;
				particles[j].ay += dy * force;
		    }

		}
		else{  // must recurse,
			treeForce(particles, j, nn.children[0], forceType) ;
			treeForce(particles, j, nn.children[1], forceType) ;
			treeForce(particles, j, nn.children[2], forceType) ;
			treeForce(particles, j, nn.children[3], forceType) ;
		}

	}
	// else - do nothing, as node has no particle or children

}




void move( particle_t *p, int type ){
	/*
	Integrate the ODE.
    Slightly simplified Velocity Verlet integration.
    Conserves energy better than explicit Euler method.

    Input:
    particle_t *p        Address of particle to move.
    int type             Force type. If using force type 1, bounce off walls.

    Output:
    Velocity and position of current particle is updated.
     */
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    if( type == 1 ){
    	// bounce from walls
		// skip for gravity type forces
		while (p->x < 0 || p->x > size) {
			p->x = p->x < 0 ? -p->x : 2 * size - p->x;
			p->vx = -p->vx;
		}

		while (p->y < 0 || p->y > size) {
			p->y = p->y < 0 ? -p->y : 2 * size - p->y;
			p->vy = -p->vy;
		}
    }
}
