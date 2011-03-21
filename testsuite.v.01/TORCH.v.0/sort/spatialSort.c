
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

/*
 Sort of spatial data (particles) according to a Z-order scheme
 
 Alex Kaiser, LBNL, 4/16/10.
*/ 

// particle data structure
typedef struct particle {
	double x;
	double y;
	unsigned int key;
} particle_t;

// structure to keep track of grid layouts
typedef struct gridData {
	double minX, minY;
	double maxX, maxY;
	int length; // must be power of two
} grid;


// headers

// sort routines
void swap(particle_t *particles, int i, int j);
void quickSort(particle_t *particles, int left, int right);

// particle and grid utilities
void init_particles(int n, particle_t *p);
grid initGrid(particle_t *particles, int n, int length);
void assignKeys(particle_t *particles, int n, grid myGrid);
grid computeCenterAndRadius(particle_t *particles, int n, grid myGrid);
void assignIndividualKey(particle_t *particle, int gridX, int gridY, grid myGrid) ; 

// random number generators
typedef struct {
	double x[2];
}dd_real;

dd_real ddSub(dd_real a, dd_real b) ;
dd_real ddMult(double a, double b);
double expm2(double p, double modulus);
double bcnrand( );

// timer
double read_timer( );

// verification 
char verifyMortonOrder(particle_t *particles, int left, int right, int level, double minX, double maxX, double minY, double maxY) ;
char verifySorted(particle_t *particles, int size);



int main() {
	
	printf("Spatial Sort test:\n");

	char allPass = 1; 
	
	int size = 1000000;
	particle_t *particles = (particle_t*) malloc(size * sizeof(particle_t));
	init_particles(size, particles);
	

	double startTime, endTime; 
	startTime = read_timer(); 
	
	// grid length must be power of two
	int length = 16;
	grid myGrid = initGrid(particles, size, length);

	// assign keys for sorting 
	assignKeys(particles, size, myGrid);

	// sort contents
	quickSort(particles, 0, size - 1);
	
	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;

	// output information on some range of particles if desired for debugging 
	/*
	int i;
	for (i = size-10; i < size; i++) {
		printf("par[%d].x = %f\n", i, particles[i].x);
		printf("par[i].gridX = %d\n", particles[i].gridX);
		printf("par[i].y = %f\n", particles[i].y);
		printf("par[i].gridY = %d\n", particles[i].gridY);
		printf("key = %x\n\n", particles[i].key);
	}
	*/ 
	
	char spatialVerificationPassed = verifyMortonOrder(particles, 0, size-1, (int) log2(length), myGrid.minX, myGrid.maxX, myGrid.minY, myGrid.maxY); 

	if( spatialVerificationPassed )
		printf("Particles arranged in Morton order.\nGeometric test passed. \n");
	else{
		fprintf(stderr, "Particles not in Morton order.\nGeometric test failed. \n");
		allPass = 0; 
	}

	if (verifySorted(particles, size))
		printf("Keys sorted properly.\nTest of key order passed. \n");
	else{
		fprintf(stderr, "Keys not in ascending order.\nTest of key order failed. \n");
		allPass = 0; 
	}
	
	if (allPass) {
		printf("\nSpatial sort test passed.\n\n\n");
	}
	else {
		fprintf(stderr, "\nSpatial sort test failed.\n\n\n"); 
	}

	free(particles) ; 
	
	return allPass;
}


void init_particles(int n, particle_t *p) {
	/*
	 Initialize the particle positions
	
	 Input:
		int n			Number of particles to initialize. 
		particle_t *p	Particle array. Must be pre-allocated to size n. 
	 
	 Output: 
		particle_t *p	Particle array, now with positions initialized. 
	 
	 */ 

	double coeff = 1.0;

	int i;
	for (i = 0; i < n; i++) {
		p[i].x = coeff * ( 2 * (bcnrand())  - 1 );
		p[i].y = coeff * ( 2 * (bcnrand())  - 1 );
	}
}

grid initGrid(particle_t *particles, int n, int length) {
	/*
	 Initialize grid parameters for spatial sorting
	 
	 Input:
		particle_t *particles		Particle array with positions initialized.
		int n						Number of particles.
		int length					Number of bins in each dimension of grid.
										Must be a power of two. 
	 
	 Output: 
		grid (returned)				Grid with parameters appropriate for current
										spatial distribution of particles
	 */ 
	
	grid myGrid;
	myGrid = computeCenterAndRadius(particles, n, myGrid);
	myGrid.length = length;

	if (length != (0x1 << (int) log2(length))) {
		fprintf(stderr, "Grid length must be a power of two. Exiting.\n");
		exit(-1);
	}

	return myGrid;
}

void assignKeys(particle_t *particles, int n, grid myGrid) {
	/* 
	 Assign grid numbers for spatial sorting of particles accoriding to 
		the grid position in which each particle lies. 
	 
	 Input:
		particle_t *particles		Particle array with positions initialized.
		int n						Number of particles.
		grid myGrid					Grid of sorting parameters
	 
	 Output:
		particle_t *particles		Particle array with keys initialized.
	 */ 
	
	int i;

	double xCoeff = (myGrid.maxX - myGrid.minX) / (double) myGrid.length;
	double yCoeff = (myGrid.maxY - myGrid.minY) / (double) myGrid.length;
	
	int gridX, gridY ; 

	// check that unsigned ints have enough bits to store keys for the requested depth
	if ((myGrid.length * 2) > (8 * sizeof(unsigned int))) {
		fprintf(stderr, "Too many bins. Number of bins must not exceed half the size of unsigned int in bits.");
		return;
	}
	
	for (i = 0; i < n; i++) {

		gridX = (int) floor((particles[i].x - myGrid.minX) / xCoeff);
		gridY = (int) floor((particles[i].y - myGrid.minY) / yCoeff);

		// should never be called 
		if (gridX == myGrid.length || gridY == myGrid.length) {
			fprintf(stderr, "errors! can't put a particle in the highest spot! \n");
			exit(-1) ; 
		}
		
		assignIndividualKey(particles+i, gridX, gridY, myGrid) ;  
	}
}

grid computeCenterAndRadius(particle_t *particles, int n, grid myGrid) {
	/* 
	 Calculates current working region according to geometry of particles. 
	 Stores parameters in grid. 

	Input:
		particle_t *particles		Particle array with positions initialized.
		int n						Number of particles.
		grid myGrid					Grid of sorting parameters, geometric parameters initialized
	 
	Output:
		grid (returned)				Grid of sorting parameters according to current 
										spatial distribution of particles
	 */ 
	
	// initialize to position of the first particle
	myGrid.minX = myGrid.maxX = particles[0].x; 
	myGrid.minY = myGrid.maxY = particles[0].y;

	int i;
	for (i = 1; i < n; i++) {
		if (particles[i].x < myGrid.minX)
			myGrid.minX = particles[i].x;

		if (particles[i].x > myGrid.maxX)
			myGrid.maxX = particles[i].x;

		if (particles[i].y < myGrid.minY)
			myGrid.minY = particles[i].y;

		if (particles[i].y > myGrid.maxY)
			myGrid.maxY = particles[i].y;
	}

	// manually adjust the max up slightly to avoid boundary problems
	myGrid.maxX += 1e-10 ;
	myGrid.maxY += 1e-10 ;

	return myGrid;
}

void assignIndividualKey(particle_t *particle, int gridX, int gridY, grid myGrid) {
	/*
	 Assign key to a particle for Morton ordering by interleaving bits. 
	 myGrid.length must be <= 8 * sizeof(unsigned int). 

	 Input:
		particle_t *particle		Pointer to particle with position initialized.
		int gridX					Grid index in X direction.
		int gridY					Grid index in Y direction.
		int n						Number of particles.
		grid myGrid					Grid of sorting parameters
	 
	 Output:
		particle_t *particle		Particle with key initialized.
	 */ 

	int bit;

	// calculate the key for current particle
	particle->key = 0;
	for (bit = (int) log2(myGrid.length); bit >= 0; bit--) {
		// take the bit at position 'bit' and shift it over to position '2*bit'
		particle->key |= (gridX & (0x1 << bit)) << bit;
		// take the bit at position 'bit' and shift it over to position '2*bit + 1'
		particle->key |= (gridY & (0x1 << bit)) << (bit + 1);
	}
}


void quickSort(particle_t *particles, int left, int right) {
	/* 
	 Simple quick sort 
	 Taken directely from "The C Programming Language", Kernighan and Ritchie.
	 Sort particles[left]...particles[right] into increasing 
		order according to their keys. 
	 
	 Input:
		particle_t *particles		Array of particles to sort on keys
		int left					First index of array to sort
		int right					Last index of array to sort
	 
	 Output:
		particle_t *particles		Array of particles sorted in the appropriate region
	 */ 

	int i, last;

	if (left >= right) /* do nothing if array contains */
		return; /* fewer than two elements */

	swap(particles, left, (left + right) / 2); /* move partition elem */
	last = left; /* to v[0] */

	for (i = left + 1; i <= right; i++) /* partition */
		if (particles[i].key < particles[left].key)
			swap(particles, ++last, i);

	swap(particles, left, last); /* restore partition  elem */

	quickSort(particles, left, last - 1);
	quickSort(particles, last + 1, right);
}


void swap(particle_t *particles, int i, int j) {
	/*   
	 Interchange v[i] and v[j] in place
	 
	 Input:
		particle_t *particles		Array of particles to sort on keys
		int i						First value to swap
		int j						Last value to swap
	 
	 Output:
		particle_t *particles		Array of particles with specified entries swapped
	 */ 
	
	particle_t temp;
	temp = particles[i];
	particles[i] = particles[j];
	particles[j] = temp;
}


char verifyMortonOrder(particle_t *particles, int left, int right, int level, double minX, double maxX, double minY, double maxY){

	/*
	 Checks that particles are in a Z order according to dimensions given. 
	
	 Arrangement of particles is as follows.
	 2 3
	 0 1
	 
	 Scheme proceeds as follows:
	  - Checks that the first group of particles are in the zero quadrant. 
	  - Advances to next quadrant once a single particle from the quadrant has been found.
	  - Fails if a particle from a previous quadrant is found. 
	  - Runs the same checks and comparisons on each quadrant recursively if desired. 
	 
	 By convention, a particle precisely on the boundary of two regions goes to right and up.
	 
	 Input: 
		particle_t *particles		Array of particles to check 
		int left					Left-most index to check (inclusive)
		int right					Right-most index to check (inclusive)
		int level					Number of additional levels to recursively check
		double minX					Lower boundary of current region in x direction
		double madX					Upper boundary of current region in x direction
		double minY					Lower boundary of current region in y direction
		double maxY					Upper boundary of current region in y direction

	 Output:
		char (returned)				Whether all particles are sorted 
	 
	 */ 

	// once a single particle crosses the boundary, all particles remaining particles must be in the next appropriate quadrant
	char pass = 1;

	int j;
	double centerX = (maxX + minX) / 2.0 ;
	double centerY = (maxY + minY) / 2.0 ;

	int numFound[4] = {0, 0, 0, 0} ;


	// check that on this level, all particles are arranged properly.
	int quad = 0 ;

	for (j = left; j <= right; j++) {

		if( quad == 0 ){
			if( particles[j].x < centerX && particles[j].y < centerY )
				numFound[0] ++ ; // we're good, continue on
			else if( (particles[j].x >= centerX) && (particles[j].y < centerY) )
				quad = 1; // advance the current quadrant
			else if( (particles[j].x < centerX) && (particles[j].y >= centerY) )
				quad = 2; 
			else if( (particles[j].x >= centerX) && (particles[j].y >= centerY) )
				quad = 3; 
			else
				fprintf(stderr, "Improper boundaries or other errors.\n");
		}

		if( quad == 1 ){
			if( particles[j].x < centerX && particles[j].y < centerY )
				return 0; // particle lies in a previous quadrant, return failure
			else if( (particles[j].x >= centerX) && (particles[j].y < centerY) )
				numFound[1]++; // continue
			else if( (particles[j].x < centerX) && (particles[j].y >= centerY) )
				quad = 2; // advance the current quadrant
			else if( (particles[j].x >= centerX) && (particles[j].y >= centerY) )
				quad = 3; 
			else
				fprintf(stderr, "Improper boundaries or other errors.\n");
		}

		if( quad == 2 ){
			if( particles[j].x < centerX && particles[j].y < centerY )
				return 0;
			else if( (particles[j].x >= centerX) && (particles[j].y < centerY) )
				return 0;
			else if( (particles[j].x < centerX) && (particles[j].y >= centerY) )
				numFound[2]++;
			else if( (particles[j].x >= centerX) && (particles[j].y >= centerY) )
				quad = 3;
			else
				fprintf(stderr, "Improper boundaries or other errors.\n");
		}

		if( quad == 3 ){
			if( particles[j].x < centerX && particles[j].y < centerY )
				return 0 ;
			else if( (particles[j].x >= centerX) && (particles[j].y < centerY) )
				return 0;
			else if( (particles[j].x < centerX) && (particles[j].y >= centerY) )
				return 0;
			else if( (particles[j].x >= centerX) && (particles[j].y >= centerY) )
				numFound[3]++;
			else
				fprintf(stderr, "Improper boundaries or other errors.\n");
		}

	}

	// check whether the lower levels are also sorted properly.
	if( level > 1){
		right = left + numFound[0] - 1; 
		if(numFound[0] > 1)
			pass &= verifyMortonOrder(particles, left, right, level-1, minX, centerX, minY, centerY) ;
		
		left += numFound[0] ; 
		right += numFound[1] ; 
		if(numFound[1] > 1)
			pass &= verifyMortonOrder(particles, left, right, level-1, centerX, maxX, minY, centerY) ;
		
		left += numFound[1] ; 
		right += numFound[2] ; 
		if(numFound[2] > 1)
			pass &= verifyMortonOrder(particles, left, right, level-1, minX, centerX, centerY, maxY) ;
		
		left += numFound[2] ; 
		right += numFound[3] ; 
		if(numFound[3] > 1)
			pass &= verifyMortonOrder(particles, left, right, level-1, centerX, maxX, centerY, maxY) ;
	}

	return pass;

}


// simple comparison that list is sorted according to its keys.
char verifySorted(particle_t *particles, int size) {
	/*
	 Simple check that list is sorted according to keys. 
	 
	 Input:
		particle_t *particles		Array of particles to sort on keys
		int size					Number of elements to check 
	 
	 Output:
		char (returned)		Whether keys are sorted. 
	 
	 */ 
	int i;
	for (i = 0; i < size - 1; i++) {
		if (particles[i].key > particles[i + 1].key)
			return 0;
	}
	return 1;
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
	 */
	
	static char initialized = 0;
	
	// variables
	static double d1, d2, d3;
	static dd_real dd1, dd2, dd3;
	
	// constants
	static double threeToTheThirtyThree ;
	static double reciprocalOfThreeToThirtyThree ;
	static double twoToTheFiftyThree ;
	
	static double startingIndex ;
	
	static unsigned long long int n;
	
	if( !initialized ){
		
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
		
		initialized = 1;
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

