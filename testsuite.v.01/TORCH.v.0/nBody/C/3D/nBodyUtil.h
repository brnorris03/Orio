

#include <stdio.h>

/*
 Utility routines for N Body calculations.

 Alex Kaiser, LBNL, 10/2010.
 */


// particle data structure
typedef struct particle{
	double x;
	double y;
	double z;
	double vx;
	double vy;
	double vz;
	double ax;
	double ay;
	double az;
	double mass;
} particle_t;

typedef struct {
	double **xForces;
	double **yForces;
	double **zForces;
	int length;
	int steps;
} forceLog;

typedef struct {
	unsigned int numParticles;
	unsigned int capacity; // this many particles have been allocated
	unsigned int *parts; // holds the indices of the bins
} bin;


// treenode data-structure
typedef struct nodeBlank{
	char isInternal;

	// info about the force recursion
	double totalMass;
	double centerMass[3];

	// info about the location and size of the recursion
	double radius;
	double center[3];

	struct nodeBlank * children;

	int particleIndex; // which particle lives at this node, by array index

	// nb:
	// if( particleIndex == -1 ), the node has no particle
	// The user may wish to move particles around to improve locality
}node;


typedef struct {
	double x[2];
}dd_real;

double size;

// constants
#define density 0.0005
#define MASSDEF    0.4
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

#define THETA 0.001   // tolerance constant for force cutoff


double read_timer( );

void save( FILE *f, int n, particle_t *p );

// cutoff and binning related
bin * initBins(int * numBins);
bin * assignBins(particle_t *particles, bin *bins, int n);
bin * cleanBins(bin *bins, int numBins);


// tree functions and utils for Barnes Hut
node * octTreeBuild(particle_t *particles, int n);
node octTreeInsert(particle_t *particles, int j, node root);
node * allocAndInitChildren(node parent);
node computeCMandTM(node nn, particle_t *particles);
node computeCenterAndRadius(node root, particle_t *particles, int n);
void freeDescendents( node nn);


forceLog allocAndInitForceLog(int length, int steps) ;
void freeForceLog(forceLog currentLog) ;



double ** allocDouble2d(int m, int n) ;
void freeDouble2d(double **z, int m, int n) ;
void printVector(double *z, int n) ;

double rmsError(double *guess, double *true, int n) ;

char verifyNBody(forceLog guess, forceLog true, double rmsErrorTol) ;


// random number generator
void resetBcnRand() ;
dd_real ddSub(dd_real a, dd_real b) ;
dd_real ddMult(double a, double b) ;
double expm2(double p, double modulus) ;
double bcnrand( ) ;
