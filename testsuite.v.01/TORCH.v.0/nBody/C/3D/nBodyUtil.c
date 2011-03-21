

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "nBodyUtil.h"

/*
 Utility routines for N Body calculations.

 Alex Kaiser, LBNL, 10/2010.
 */



void save( FILE *f, int n, particle_t *p ){
	/*
	 Prints position of particles to supplied file pointer.

	 Input:
	 FILE *f           File for writing.
	 int n             Number of particles to output.
	 particle_t *p     Particles to print.

	 Output:
	 Positions of particles are written to supplied file.
	 */

    static char first = 1;
    if( first )
    {
		// print an extra zero to make 3 columns for easy verification
        fprintf( f, "%d %g %d\n", n, size, 0 );
        first = 0;
    }

    int i;
    for(i=0; i<n; i++)
        fprintf( f, "%.16g %.16g %.16g\n", p[i].x, p[i].y, p[i].z );
}


double read_timer( ){
	/*
	 Timer.
	 Returns elapsed time since initialization as a double.

	 Output:
	 double (returned)   Elapsed time.
	 */

    static char initialized = 0;
    static struct timeval start;
    struct timeval end;

    if( !initialized ){
        gettimeofday( &start, NULL );
        initialized = 1;
    }

    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}



bin * initBins(int * numBins){
	/*
	 Binning code for cutoff method.

	 Note:
	     size is global.
	     cutoff is a define.

	 Input:
	 int *numBins         Outputs the number of bins allocated.
	                          Set here.

	 Output:
	 bin * (returned)     Array of bins for cutoff method.
	 int *numBins         Length of the above array.
	 */

	*numBins = ceil(size / (cutoff*cutoff) ) ;
	int i;

	bin *bins = (bin *) malloc( (*numBins) * sizeof(bin)) ;

	for(i=0; i < (*numBins); i++){
		bins[i].numParticles = 0;
		bins[i].capacity = 100; // tunable, amount to initially allocated
		bins[i].parts = (unsigned int *) malloc(bins[i].capacity * sizeof(unsigned int)) ;
	}
	return bins;
}


bin * assignBins(particle_t *particles, bin *bins, int n){
	/*
	 Assigns particles to bins for cutoff method.

	 Input:
	 particle_t *particles       Particles to assign.
	 bin *bins                   Bins in which to place particles.
	                                 Must be pre-initialized by calling initBins.
	 int n                       Number of particles.

	 Output:
	 bin *bins (returned)        Bins with particles set.
	 */

	int binNum;
	int i, k;
	for(i=0; i<n; i++){
		binNum = ceil( particles[i].x / cutoff*cutoff ) - 1 ;

		// handle the (extremely rare) case that x is exactly zero
		if(binNum < 0) binNum=0;

		//resize array of indices if necessary
		if( bins[binNum].capacity <= bins[binNum].numParticles){
			bins[binNum].parts = (unsigned int *) realloc(bins[binNum].parts, 2 * bins[binNum].capacity * sizeof(unsigned int) );
			bins[binNum].capacity *= 2;
		}

		k = bins[binNum].numParticles ;
		bins[binNum].parts[k] = i; // adds current particle to the particle list
		bins[binNum].numParticles ++ ;
	}
	return bins;
}


bin * cleanBins(bin *bins, int numBins){
	/*
	 Removes all particles from bins.

	 Input:
	 bin *bins                   Bins in which to place particles.
	                                 Must be pre-initialized by calling initBins.
	                                 Does not modify particle pointers, only numParticles.
	                                 Use care that this parameter is correctly modified.
	 int numBins                 Number of bins.

	 Output:
	 bin *bins (returned)        Bins with particles removed.
	 */
	int i;

	for(i=0; i<numBins; i++){
		bins[i].numParticles = 0;
	}
	// don't bother cleaning the parts array, but use the numParticles carefully
	return bins;
}


node * octTreeBuild(particle_t *particles, int n){
	/*
	 Builds an adaptive octtree according to the given distribution of particles.

	 Input:
	 particle_t *particles        Particles to place into tree.
	 int n                        Number of particles.

	 Output:
	 node * (returned)            Pointer to root node of the quadtree.
	 */

	node *rootPtr = (node *) malloc(sizeof(node)) ; // just one, others added recursively
	// must malloc so local var. isn't killed.

	if (rootPtr == NULL){
		printf("malloc error. null pointer\n");
		exit(1);
	}

	*rootPtr = computeCenterAndRadius(*rootPtr, particles, n);

	rootPtr[0].particleIndex = -1;  // initial values so recursion begins correctly
	rootPtr[0].isInternal = 0;

	int j;
	for(j=0; j<n; j++){
		*rootPtr = octTreeInsert(particles, j, *rootPtr);
	}

	// process empty leaves if desired here.

	return rootPtr;
}


node octTreeInsert(particle_t *particles, int j, node root){
	/*
	 Inserts a particle into an adaptive octtree.

	 Input:
	 particle_t *particles        Array of particles.
	 int j                        Index of particle to place.
	 node root                    Root of current subtree.

	 Output:
	 node * (returned)            Pointer to root node of current subtree.
	 */


	int index ;

	if(root.isInternal){  // root has 8 children

		// calculate which child of root should be interacted with
		// each x-y slice numbered from lower left, counter clockwise
		// add 4 to index to increase z slice
		index = 0;
		if(root.center[0] < particles[j].x ){
			index = 1;
		}
		if(root.center[1] < particles[j].y ){
			if(index == 0)
				index = 3;
			else
				index = 2;
		}
		if(root.center[2] < particles[j].z ){  // for z direction, simply add 4
			index += 4;
		}


		// add the node to appropriate child
		root.children[index] = octTreeInsert(particles, j, root.children[index] );
	}

	else if( root.particleIndex != -1){ // root is a leaf - it has a particle and so no children
		root.isInternal = 1; // node is now internal, current particle will be bubbled down

		// allocate children to current node
		root.children = allocAndInitChildren( root );

		// determine where to move current particle (the one already at this node)
		index = 0;
		int currentParticleIndex = root.particleIndex ;
		if(root.center[0] < particles[currentParticleIndex].x ){
			index = 1;
		}
		if(root.center[1] < particles[currentParticleIndex].y ){
			if(index == 0)
				index = 3;
			else
				index = 2;
		}
		if(root.center[2] < particles[currentParticleIndex].z ){
			index += 4;
		}

		// and move it to the appropriate child
		root.children[index].particleIndex = currentParticleIndex ;

		// this node no longer contains a particle, set index back to -1
		root.particleIndex = -1;

		// old particle at this node has been bubbled down
		// calculate which child of root the NEW particle should be interacted with
		index = 0;
		if(root.center[0] < particles[j].x ){ // move particle right if passes
			index = 1;
		}
		if(root.center[1] < particles[j].y ){ // move up if passes
			if(index == 0)
				index = 3;
			else
				index = 2;
		}
		if(root.center[2] < particles[j].z ){ // move up if passes
			index += 4;
		}

		// add the new particle at the appropriate child
		root.children[index] = octTreeInsert(particles, j, root.children[index]) ;
	}

	else{ // root is empty - put the particle there.
		root.particleIndex = j;
	}

	return root;
}



void freeDescendents( node nn ){
	/*
	 Recursively frees all descendants of a node
	 Does not touch node itself.

	 Input:
	 node nn      Root of current subtree.
	 */
	if (nn.isInternal){
		int i;

		for(i=0; i<8; i++){
			if(nn.children[i].isInternal)
				freeDescendents(nn.children[i]);
		}

		free( nn.children) ;
	}
}

node computeCenterAndRadius(node root, particle_t *particles, int n){
	/*
	 Calculates boundaries and center of the current region.

	 Input:
	 node root                    Root node of tree of particles.
	 particle_t *particles        Array of particles.
	 int n                        Number of particles.

	 Output:
	 node root                    Root node of tree of particles.
	                                  Boundaries initialized.
	 */

	double minX, minY, minZ;
	double maxX, maxY, maxZ;


	minX = maxX = particles[0].x; // initialize to position of the first particle
	minY = maxY = particles[0].y;
	minZ = maxZ = particles[0].z;


	int i;
	for(i=1; i<n; i++){
		if(particles[i].x < minX)
			minX = particles[i].x;

		if(particles[i].x > maxX)
			maxX = particles[i].x;

		if(particles[i].y < minY)
			minY = particles[i].y;

		if(particles[i].y > maxY)
			maxY = particles[i].y;

		if(particles[i].z < minZ)
			minZ = particles[i].z;

		if(particles[i].z > maxZ)
			maxZ = particles[i].z;
	}

	root.radius = .5 * (maxX-minX);
	if( (maxY-minY) > root.radius ){
		root.radius = .5 * (maxY-minY);
	}
	if( (maxZ-minZ) > root.radius ){
		root.radius = .5 * (maxZ-minZ);
	}

	root.center[0] = .5 * (maxX + minX);
	root.center[1] = .5 * (maxY + minY);
	root.center[2] = .5 * (maxZ + minZ);

	return root;
}


node * allocAndInitChildren(node parent){
	/*
	 Allocates children of parent node.

	 Input:
	 node parent                  Parent node.

	 Output:
	 node *children (returned)    Array of the four children with boundaries initialized.
	 */

	node * children = (node *) malloc(8 * sizeof(node)) ;

		if (children == NULL){
			printf("malloc error. null pointer\n");
			exit(1);
		}

		double rad = parent.radius / 2.0;  // radius of children nodes

		// init children to proper values around parent
		// newly initialized children have no particles, and are not internal nodes

		// lower left
		children[0].center[0] = parent.center[0] - rad;
		children[0].center[1] = parent.center[1] - rad;
		children[0].center[2] = parent.center[2] - rad;
		children[0].radius = rad;
		children[0].particleIndex = -1;
		children[0].isInternal = 0;

		// lower right
		children[1].center[0] = parent.center[0] + rad;
		children[1].center[1] = parent.center[1] - rad;
		children[1].center[2] = parent.center[2] - rad;
		children[1].radius = rad;
		children[1].particleIndex = -1;
		children[1].isInternal = 0;

		// upper right
		children[2].center[0] = parent.center[0] + rad;
		children[2].center[1] = parent.center[1] + rad;
		children[2].center[2] = parent.center[2] - rad;
		children[2].radius = rad;
		children[2].particleIndex = -1;
		children[2].isInternal = 0;

		// upper left
		children[3].center[0] = parent.center[0] - rad;
		children[3].center[1] = parent.center[1] + rad;
		children[3].center[2] = parent.center[2] - rad;
		children[3].radius = rad;
		children[3].particleIndex = -1;
		children[3].isInternal = 0;

		// second half
		// lower left
		children[4].center[0] = parent.center[0] - rad;
		children[4].center[1] = parent.center[1] - rad;
		children[4].center[2] = parent.center[2] + rad;
		children[4].radius = rad;
		children[4].particleIndex = -1;
		children[4].isInternal = 0;

		// lower right
		children[5].center[0] = parent.center[0] + rad;
		children[5].center[1] = parent.center[1] - rad;
		children[5].center[2] = parent.center[2] + rad;
		children[5].radius = rad;
		children[5].particleIndex = -1;
		children[5].isInternal = 0;

		// upper right
		children[6].center[0] = parent.center[0] + rad;
		children[6].center[1] = parent.center[1] + rad;
		children[6].center[2] = parent.center[2] + rad;
		children[6].radius = rad;
		children[6].particleIndex = -1;
		children[6].isInternal = 0;

		// upper left
		children[7].center[0] = parent.center[0] - rad;
		children[7].center[1] = parent.center[1] + rad;
		children[7].center[2] = parent.center[2] + rad;
		children[7].radius = rad;
		children[7].particleIndex = -1;
		children[7].isInternal = 0;

		return children;
}


node computeCMandTM(node nn, particle_t *particles){
	/*
	 Computes the center and total mass of an entire tree.
	 Processes internal nodes recursively.

	 Input:
	 node nn                      Node of which to compute center and total mass.
	 particle_t *particles        Array of particles.

	 Output:
	 node nn (returned)           Node with updated CM and TM
	 */

	if(nn.particleIndex != -1){  // if n is a leaf
		nn.centerMass[0] = particles[nn.particleIndex].x ;
		nn.centerMass[1] = particles[nn.particleIndex].y ;
		nn.centerMass[2] = particles[nn.particleIndex].z ;
		nn.totalMass = particles[nn.particleIndex].mass ;
	}
	else if(nn.isInternal){

		// post order traversal
		// fully process all children
		int i, j;
		for(i=0; i<8; i++){
			nn.children[i] = computeCMandTM(nn.children[i], particles);
		}

		// total mass is mass of all children
		nn.totalMass = 0.0;
		for(i=0; i<8; i++){
			nn.totalMass += nn.children[i].totalMass ;
		}

		if(nn.totalMass == 0.0){ // just in case, should never occur if recursion works properly
			printf("big problems. zero mass somewhere\n") ;
			exit(1);
		}

		// each coordinate of center of mass is mass weighted sum of all children's
		for(j=0; j<3; j++){
			nn.centerMass[j] = 0.0;
			for(i=0; i<8; i++){
				nn.centerMass[j] += nn.children[i].totalMass * nn.children[i].centerMass[j] ;
			}
			nn.centerMass[j] /= nn.totalMass ;
		}

	}
	else{
		// nn is a node with nothing, set it's values to zero
		// this handles a bunch of generality but can't be good for performance
		// shouldn't keep visiting empty nodes
		nn.centerMass[0] = 0.0;
		nn.centerMass[1] = 0.0;
		nn.centerMass[2] = 0.0;
		nn.totalMass = 0.0;

	}


	return nn;
}



forceLog allocAndInitForceLog(int length, int steps){
	/*
	 Initializes and allocates space for logging forces.
	 Used for keeping verification information.

	 Input:
	 int length       Number of particles on which to keep information.
	 int steps        Number of time steps for which to keep information.

	 Output:
	 forceLog         Structure forceLog, with arrays allocated to (steps * length)
	 */

	forceLog toReturn;
	toReturn.length = length ;
	toReturn.steps = steps ;
	toReturn.xForces = allocDouble2d(steps, length) ;
	toReturn.yForces = allocDouble2d(steps, length) ;
	toReturn.zForces = allocDouble2d(steps, length) ;

	return toReturn ;
}

void freeForceLog(forceLog currentLog){
	/*
	 Frees arrays of a forceLog.
	 Does not touch structure itself.

	 Input:
	 forceLog currentLog     Structure of which to free arrays.
	 */

	freeDouble2d(currentLog.xForces, currentLog.steps, currentLog.length) ;
	freeDouble2d(currentLog.yForces, currentLog.steps, currentLog.length) ;
	freeDouble2d(currentLog.zForces, currentLog.steps, currentLog.length) ;
}


double ** allocDouble2d(int m, int n){
	/*
	 Returns ptr to and (m by n) array of double precision real

	 Input:
	 int m,n                  Matrix dimensions.

	 Output:
	 double ** (returned)     m by n double precision array
	 */

	double ** temp = (double **) malloc( m * sizeof(double *) ) ;
	int j;
	for(j=0; j<m; j++){
		temp[j] = (double *) malloc( n * sizeof(double) ) ;
	}
	return temp;
}


void freeDouble2d(double **z, int m, int n){
	/*
	 Frees (m by n) array of double precision reals.

	 Input:
	 double **z                Array to free.
	 int m, n                  Matrix dimensions.
	 */
	int j;
	for(j=0; j<m; j++){
		free( z[j] );
	}
	free(z) ;
}

void printVector(double *z, int n){
	/*
	 Prints double array to stdout.

	 Input:
		double * z			Array to print.
		int n				Length.
	 */

	int i;
	for(i=0; i<n; i++)
		printf("%f ", z[i]) ;

	printf("\n");
}


double rmsError(double *guess, double *true, int n){
	/*
	 Returns rms error between double precision vectors.

	 Input:
	 double *guess, *true      Vectors to compare.
	 int n                     Their length.

	 Output:
	 double (returned)         RMS error between the two vectors.
	 */
	double err = 0.0;

	int k;
	for(k=0; k<n; k++){
		err += (guess[k] - true[k]) * (guess[k] - true[k]);
	}

	return sqrt( err / (double) n) ;
}


char verifyNBody(forceLog guess, forceLog true, double rmsErrorTol){
	/*
	 Verifies that the force log guess is accurate to within the given tolerance at each time step.
	 Checks the first true.steps values of guess.

	 Input:
	 forceLog guess         Log of forces from method to verify.
	 forceLog true          Log of forces from naive method, or other method believed to be correct.
	 double rmsErrorTol     Tolerance for RMS error.

	 Output:
	 char pass (returned)   Whether RMS error in each dimension is under tolerance.
	 */

	if( guess.steps < true.steps){
		fprintf(stderr, "Guess array must have at least as many time steps logged as true.\nTest failed.\n\n") ;
		return 0;
	}
	if( guess.length != true.length ){
		fprintf(stderr, "Both input arrays must have the same length at each time step.\nTest failed.\n\n") ;
	}

	double currentErrX, currentErrY, currentErrZ ;
	double maxErrX = 0.0 ;
	double maxErrY = 0.0 ;
	double maxErrZ = 0.0 ;

	int i;
	for(i=0; i<true.steps; i++){

		currentErrX = rmsError( guess.xForces[i], true.xForces[i], true.length) ;
		if(currentErrX > maxErrX)
			maxErrX = currentErrX ;

		currentErrY = rmsError( guess.yForces[i], true.yForces[i], true.length) ;
		if(currentErrY > maxErrY)
			maxErrY = currentErrY ;

		currentErrZ = rmsError( guess.zForces[i], true.zForces[i], true.length) ;
		if(currentErrZ > maxErrZ)
			maxErrZ = currentErrZ ;
	}

	printf("Maximum RMS error in X forces = %.10e\n", maxErrX ) ;
	printf("Maximum RMS error in Y forces = %.10e\n", maxErrY ) ;
	printf("Maximum RMS error in Z forces = %.10e\n\n", maxErrZ ) ;

	if( (maxErrX < rmsErrorTol) && (maxErrY < rmsErrorTol))
		return 1;

	return 0;
}


static char bcnRandIsInitialized = 0;

void resetBcnRand(){
	/*
	 Resets flag for bcnRand.
	 Re-initializes generator.
	 */
	bcnRandIsInitialized = 0;
}

double bcnrand( ){
	/*
	 This routine generates a sequence of IEEE 64-bit floating-point pseudorandom
	 numbers in the range (0, 1), based on the recently discovered class of
	 normal numbers described in the paper "Random Generators and Normal Numbers"
	 by David H. Bailey and Richard Crandall, available at http://www.nersc.gov/~dhbailey.

	 Internal, private variables are allocated as static and should not be modified by the user.
	 User should simply call the routine for each random number desired.

	 Parameter:
	 static double startingIndex       Users may modify this parameter to control where in the sequence the generator starts.
	                                   This facility is useful for parallel implementations of this generator
	                                   to ensure that each processor gets the correct portion of the sequence.

	 Output:
	 double (returned)                 Random number in (0,1).
	 */

	// variables
	static double d1, d2, d3;
	static dd_real dd1, dd2, dd3;

	// constants
	static double threeToTheThirtyThree ;
	static double reciprocalOfThreeToThirtyThree ;
	static double twoToTheFiftyThree ;

	static double startingIndex ;

	static unsigned long long int n;

	if( !bcnRandIsInitialized ){

		n = 0;
		startingIndex = pow(3.0, 33) + 10000.0 ;

		// parameters
		threeToTheThirtyThree = pow(3, 33) ;
		reciprocalOfThreeToThirtyThree = 1.0 / threeToTheThirtyThree ;
		twoToTheFiftyThree = pow(2, 53);

		// check inputs
		if( (startingIndex < threeToTheThirtyThree + 100) || (startingIndex > twoToTheFiftyThree) ){
			fprintf(stderr, "bcnrand error: startingIndex must satisfy 3^33 + 100 <= startingIndex <= 2^53\nstartingIndex = %.15f", startingIndex);
			exit(-1) ;
		}

		// calculate starting element
		d2 = expm2(startingIndex - threeToTheThirtyThree, threeToTheThirtyThree) ;
		d3 = trunc(0.5 * threeToTheThirtyThree) ;
		dd1 = ddMult(d2, d3) ;
		d1 = trunc(reciprocalOfThreeToThirtyThree * dd1.x[0]) ;
		dd2 = ddMult(d1, threeToTheThirtyThree) ;
		dd3 = ddSub(dd1, dd2) ;
		d1 = dd3.x[0] ;

		if(d1 < 0.0)
			d1 += threeToTheThirtyThree ;

		bcnRandIsInitialized = 1;
	}

	n++ ;
	if( n > (2.0 * threeToTheThirtyThree / 3.0) ){
		fprintf(stderr, "bcnrand warning: number of elements exceeds period.\nn = %llu\nperiod = %f", n, 2.0 * twoToTheFiftyThree / 3.0) ;
	}

	// calculate next element of sequence
	dd1.x[0] = twoToTheFiftyThree * d1 ;
	dd1.x[1] = 0.0;
	d2 = trunc(twoToTheFiftyThree * d1 / threeToTheThirtyThree) ;
	dd2 = ddMult(threeToTheThirtyThree, d2) ;
	dd3 = ddSub(dd1, dd2);
	d1 = dd3.x[0];

	if(d1 < 0.0)
		d1 += threeToTheThirtyThree ;

	return reciprocalOfThreeToThirtyThree * d1 ;
}

double expm2(double p, double modulus){
	/*
	 Returns 2^p mod am.  This routine uses a left-to-right binary scheme.

	 Input:
	 double p            Exponent.
	 double modulus      Modulus.

	 Output:
	 double (returned) 2^p mod am
	 */

	double reciprocalOfModulus;
	double d2;
	dd_real dd1, dd2, dd3, ddModulus;
	double p1, pt1, r;


	double twoToTheFiftyThree = pow(2, 53);

	reciprocalOfModulus = 1.0 / modulus;
	p1 = p ;
	pt1 = twoToTheFiftyThree;
	r = 1.0 ;
	ddModulus.x[0] = modulus;
	ddModulus.x[1] = 0.0;

	while(1){

		if(p1 >= pt1){
			// r = mod (2.d0 * r, am)
			dd1 = ddMult(2.0, r) ;
			if( dd1.x[0] > modulus){
				dd2 = ddSub(dd1, ddModulus) ;
				dd1 = dd2 ;
			}
			r = dd1.x[0] ;
			p1 -= pt1 ;
		}

		pt1 *= 0.5 ;

		if(pt1 >= 1.0){
			//r = mod (r * r, am)
			dd1 = ddMult(r, r) ;
			dd2.x[0] = reciprocalOfModulus * dd1.x[0] ;
			d2 = trunc(dd2.x[0]) ;
			dd2 = ddMult(modulus, d2) ;
			dd3 = ddSub(dd1, dd2) ;
			r = dd3.x[0] ;

			if(r < 0.0)
				r += modulus ;
		}
		else
			break;
	}

	return r;
}


dd_real ddMult(double da, double db){
	/*
	 Returns res = a * b, where res is double double precision

	 Input:
	 double da, db         Values to multiply

	 Output:
	 returned (dd_real)    Their products in double double precision
	 */

	dd_real res ;

	double split = 134217729.0 ;

	double a1, a2, b1, b2, cona, conb ;

	cona = da * split ;
	conb = db * split ;
	a1 = cona - (cona - da) ;
	b1 = conb - (conb - db) ;
	a2 = da - a1 ;
	b2 = db - b1 ;

	res.x[0] = da * db ;
	res.x[1] = (((a1 * b1 - res.x[0]) + a1 * b2) + a2 * b1) + a2 * b2 ;

	return res;

}

dd_real ddSub(dd_real a, dd_real b){
	/*
	 Returns res = a - b, where res is double double precision

	 Input:
	 double da, db         Values to subtract

	 Output:
	 returned (dd_real)    Their difference in double double precision
	 */

	double e, temp1, temp2;
	dd_real res ;

	temp1 = a.x[0] - b.x[0] ;
	e = temp1 - a.x[0] ;
	temp2 = ((-b.x[0] - e) + (a.x[0] - (temp1 - e))) + a.x[1] - b.x[1] ;

	res.x[0] = temp1 + temp2 ;
	res.x[1] = temp2 - (res.x[0] - temp1) ;

	return res;
}




