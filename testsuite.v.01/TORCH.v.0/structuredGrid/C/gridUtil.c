
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gridUtil.h"

#include "complexUtil.h"

/*
 Structured grid utilities.

 Alex Kaiser, LNBL, 7/2010
 */



/*
 // use allocator from complexUtil.h because this produces a duplicate symbol
double *** allocDouble3d(int m, int n, int p){
	// returns ptr to and (m by n by p) array of double precision real
	double *** temp = (double ***) malloc( m * sizeof(double **) ) ;
	int j,k;
	for(j=0; j<m; j++){
		temp[j] = (double **) malloc( n * sizeof(double *) ) ;
		for(k=0; k<n; k++){
			temp[j][k] = (double *) malloc( p * sizeof(double));
		}
	}
	return temp;
}

void freeDouble3d(double ***z, int m, int n, int p){
	// frees (m by n by p) array of double precision reals
	int j,k;
	for(j=0; j<m; j++){
		for(k=0; k<n; k++){
			free( z[j][k] );
		}
		free( z[j] );
	}
	free(z) ;
}
*/


grid initGrid3d( double f(double xArg, double yArg, double zArg, int typeArg, double toughnessArg), double height, double x, double y, double z, int type, double toughness) {
	/*
	 Initializes, allocates and returns a structured grid according to the supplied function
	 Ghost zones included, grid initialized to (m+2, n+2, p+2).
	 Minimum value for each dimension is zero.

	 Input:
	 double f(double xArg, double yArg, double zArg)   The function, must be from R3 to reals. f:R^3 -> R
	 double h  Grid height
	 double x  Maximum x value
	 double y  Maximum y value
	 double z  Maximum z value

	 Output:
	 grid (returned) Grid structure with values initialized to the function values on mesh points
	 */

	int i,j,k;

	int m = (int) (x / height - 1);
	int n = (int) (y / height - 1);
	int p = (int) (z / height - 1);

	grid g ;

	g.m = m;
	g.n = n;
	g.p = p;
	g.height = height;

	g.values = allocDouble3d(m+2, n+2, p+2) ;

	for(i=0; i<m+2; i++){
		for(j=0; j<n+2; j++){
			for(k=0; k<p+2; k++){
				g.values[i][j][k] = f(i*height, j*height, k*height, type, toughness) ;
			}
		}
	}

	return g;
}

void setGrid3d(grid *g, double f(double xArg, double yArg, double zArg, int typeArg, double toughnessArg), int type, double toughness) {
	/*
	 Re-initializes and returns a structured grid according to the supplied function.
	 Does not allocate new memory.
	 Ghost zones included, grid initialized to (m+2, n+2, p+2).
	 Minimum value for each dimension is zero.

	 Input:
	 double f(double xArg, double yArg, double zArg)   The function, must be from R3 to reals. f:R^3 -> R

	 Output:
	 grid (returned) Grid structure with values initialized to the function values on mesh points
	 */

	int i,j,k;

	for(i=0; i<g->m+2; i++){
		for(j=0; j<g->n+2; j++){
			for(k=0; k<g->p+2; k++){
				g->values[i][j][k] = f(i*g->height, j*g->height, k*g->height, type, toughness) ;
			}
		}
	}

}

grid initZeroGrid3d(int m, int n, int p, double height){
	/*
	 Initializes, allocates and returns a structured grid with all zeros

	 Input:
	 int m,n,p  Grid dimensions

	 Output:
	 grid (returned) Grid structure with values initialized to zero on all mesh points
	 */

	int i,j,k;

	grid g ;

	g.m = m;
	g.n = n;
	g.p = p;
	g.height = height;

	g.values = allocDouble3d(m+2, n+2, p+2) ;

	for(i=0; i<m+2; i++){
		for(j=0; j<n+2; j++){
			for(k=0; k<p+2; k++){
				g.values[i][j][k] = 0.0 ;
			}
		}
	}

	return g;
}

void setZeroGrid3d(grid *g){
	/*
	 Initializes a preallocated structured grid to all zeros

	 Input:
	 grid *g       Address of grid to zero.

	 Output:
	 grid (returned) Grid structure with values initialized to zero on all mesh points
	 */

	int i,j,k;

	for(i=0; i<g->m+2; i++){
		for(j=0; j<g->n+2; j++){
			for(k=0; k<g->p+2; k++){
				g->values[i][j][k] = 0.0 ;
			}
		}
	}
}


void freeGrid(grid g){
	/*
	 Frees arrays of grid.
	 Does not modify statically allocated grid.

	 Input:
	 grid g  Grid to free
	 */

	freeDouble3d(g.values, g.m+2, g.n+2, g.p+2) ;
}



vector *** allocVector3d(int m, int n, int p){
	/*
	 Returns ptr to an (m by n by p) array of 3d vector structs

	 Input:
	 int m,n,p   Dimensions to allocated 3d vector array

	 Output:
	 vector *** (returned)  3d array of vectors
	 */

	vector ***toReturn ;

	toReturn = (vector ***) malloc( m * sizeof(vector **) ) ;
	int j,k;
	for(j=0; j<m; j++){
		toReturn[j] = (vector **) malloc( n * sizeof(vector *) ) ;
		for(k=0; k<n; k++){
			toReturn[j][k] = (vector *) malloc( p * sizeof(vector));
		}
	}
	return toReturn;
}

void freeVector3d(vector ***z, int m, int n, int p){
	// frees (m by n by p) array of 3d vector structs
	int j,k;
	for(j=0; j<m; j++){
		for(k=0; k<n; k++){
			free( z[j][k] );
		}
		free( z[j] );
	}
	free(z) ;
}


vectorField initVectorField3d( vector f(double xArg, double yArg, double zArg, int typeArg, double toughnessArg), double height, double x, double y, double z, int type, double toughness) {
	/*
	 Initializes, allocates and returns a vector field according to the supplied function.
	 Ghost zones included, grid initialized to (m+2, n+2, p+2).

	 Input:
	 vector f(double xArg, double yArg, double zArg)   The function, must be from R3 to R3. f:R^3 -> R^3
	 double h  Grid height
	 double x  Maximum x value
	 double y  Maximum y value
	 double z  Maximum z value

	 Output:
	 vectorField (returned) vectorField structure with values initialized to the function values on mesh points
	 */

	int i,j,k;

	int m = (int) (x / height - 1);
	int n = (int) (y / height - 1);
	int p = (int) (z / height - 1);

	vectorField v ;

	v.m = m;
	v.n = n;
	v.p = p;
	v.height = height;

	v.values = allocVector3d(m+2, n+2, p+2) ;

	for(i=0; i<m+2; i++){
		for(j=0; j<n+2; j++){
			for(k=0; k<p+2; k++){
				v.values[i][j][k] = f(i*height, j*height, k*height, type, toughness) ;
			}
		}
	}

	return v;
}

void setVectorField3d(vectorField *v, vector f(double xArg, double yArg, double zArg, int typeArg, double toughnessArg), int type, double toughness){
	/*
	 Initializes and returns a vector field according to the supplied function.
	 Vector field must be preallocated.

	 Input:
	 vectorField *v    Address of vector field to zero.
	 vector f(double xArg, double yArg, double zArg)   The function, must be from R3 to R3. f:R^3 -> R^3


	 Output:
	 vectorField *v  vectorField structure with values initialized to the function values on mesh points
	 */

	int i,j,k;

	for(i=0; i<v->m+2; i++){
		for(j=0; j<v->n+2; j++){
			for(k=0; k<v->p+2; k++){
				v->values[i][j][k] = f(i*v->height, j*v->height, k*v->height, type, toughness) ;
			}
		}
	}

}

vectorField initZeroVectorField3d(int m, int n, int p, double height) {
	/*
	 Initializes, allocates and returns a vector field with all values initialized to zero.
	 Ghost zones included, grid initialized to (m+2, n+2, p+2).

	 Input:
	 int m,n,p  Vector field dimensions
	 double h  Grid height

	 Output:
	 vectorField (returned) vectorField structure with values initialized to zero
	 */

	int i,j,k;

	vectorField v ;

	v.m = m;
	v.n = n;
	v.p = p;
	v.height = height;

	v.values = allocVector3d(m+2, n+2, p+2) ;

	for(i=0; i<m+2; i++){
		for(j=0; j<n+2; j++){
			for(k=0; k<p+2; k++){
				v.values[i][j][k].x = 0.0;
				v.values[i][j][k].y = 0.0;
				v.values[i][j][k].z = 0.0;
			}
		}
	}

	return v;
}

void setZeroVectorField3d(vectorField *v) {
	/*
	 Zeros a preallocated vector field.

	 Input:
	 vectorField *v   Preallocated vector field to zero.

	 Output:
	 vectorField v    vectorField structure with values set to zero.
	 */

	int i,j,k;

	for(i=0; i<v->m+2; i++){
		for(j=0; j<v->n+2; j++){
			for(k=0; k<v->p+2; k++){
				v->values[i][j][k].x = 0.0;
				v->values[i][j][k].y = 0.0;
				v->values[i][j][k].z = 0.0;
			}
		}
	}
}

void freeVectorField(vectorField f){
	/*
	 Frees arrays of grid.
	 Does not modify statically allocated grid.

	 Input:
	 vectorField f  vectorField to free
	 */

	freeVector3d(f.values, f.m+2, f.n+2, f.p+2) ;
}

void freeGridArray(gridArray currentArray){
	/*
	 Frees grid array.
	 Frees each individual grid, then frees the pointed to the array of grids.
	 Does not motify statically allocated grid array.

	 Input:
	 gridArray currentArray  Grid array to free.
	 */


	int i;
	for(i=0; i < currentArray.tSteps; i++)
		freeGrid(currentArray.grids[i]) ;

	free(currentArray.grids) ;
}

mgGridArray allocMultiGridArrays(int m, int n, int p, double height, double lambda, int depth){
	/*
	 Allocates array of mgGrids for multigrid solve.
	 Each subsequent grid has twice the grid height.
	 size(fine) = 2 * size(coarse) - 1

	 Input:
	 int m,n,p   Dimensions of most fine grid
	 double height   Height of most fine grid
	 double lambda   Lambda for most fine grid. Lambda = dt / (height * height)
	 */

	if( (0x1 << depth) >= m){
	    fprintf(stderr, "Depth must be shallow enough that grid is not divided past one element.\nExiting.\n");
	    exit(-1) ;
	}

	mgGridArray currentMgArray ;
	currentMgArray.depth = depth ;

	currentMgArray.mgGrids = (mgGrid *) malloc(depth * sizeof(mgGrid)) ;

	int i;
	for(i=0; i<depth; i++){
		currentMgArray.mgGrids[i].guess = initZeroGrid3d(m,n,p,height) ;
		currentMgArray.mgGrids[i].rhs = initZeroGrid3d(m,n,p,height) ;
		currentMgArray.mgGrids[i].residual = initZeroGrid3d(m,n,p,height) ;
		currentMgArray.mgGrids[i].lambda = lambda ;

		// adjust parameters for next grid level
		m >>= 1; // divide all lengths by two, this will truncate remaining halfs, but that's what's wanted
		n >>= 1;
		p >>= 1;
		height *= 2.0 ;
		lambda *= 0.25 ;
	}

	return currentMgArray ;
}


void gridToVector(grid g, double *x){
	/*
	 Copies grid into vector x.
	 Ignores boundaries considering homogeneous boundary conditions.
	 Uses row major ordering.

	 Input:
	 grid g			Grid to copy.
	 double *x		Output vector.

	 Output:
	 double *x		Grid contents in vector form.
	 */

	int i,j,k;
	int linearIndex = 0;

	for(i=1; i <= g.m; i++){
		for(j=1; j <= g.n; j++){
			for(k=1; k <= g.p; k++){
				x[linearIndex] = g.values[i][j][k] ;
				linearIndex++ ;
			}
		}
	}
}

void vectorToGrid(grid *g, double *x){
	/*
	 Copies vector x into grid.
	 Ignores boundaries considering homogeneous boundary conditions.
	 Uses row major ordering.

	 Input:
	 grid *g			Output grid.
	 double *x			Input vector.

	 Output:
	 grid *g			Output grid.
	 */

	int i,j,k;
	int linearIndex = 0;


	// possible that loop order here is backwards...
	for(i=1; i <= g->m; i++){
		for(j=1; j <= g->n; j++){
			for(k=1; k <= g->p; k++){
				g->values[i][j][k] = x[linearIndex] ;
				linearIndex++ ;
			}
		}
	}
}

double normGrid(grid g){
	/*
	 Calculates element-wise l2 norm of grid array.
	 Does not include ghost zones in norm calculation, but assumes they are included with the grid.

	 Input:
		grid g			Array to calculate norm.

	 Output:
	 double (returned)  Element-wise l2 norm of grid.
	 */


	double runningNorm = 0.0;
	int i,j,k;

	for(i=1; i <= g.m; i++){
		for(j=1; j <= g.n; j++){
			for(k=1; k <= g.p; k++){
				runningNorm += g.values[i][j][k] * g.values[i][j][k] ;
			}
		}
	}

	return sqrt(runningNorm);
}



double l2RelativeErrGrid(grid guess, grid trueValue){
	/*
	 Computes element-wise l2 relative error between two grids.
	 norm( guess - trueValue ) / norm(trueValue)

	 Input:
		grid guess			Guess grid.
		grid trueValue		True grid.

	 Output:
		double (returned)		l2 relative error between two input vectors.
	 */


	int i,j,k;

	grid diffs = initZeroGrid3d(trueValue.m, trueValue.n, trueValue.p, trueValue.height) ;

	for(i=1; i <= trueValue.m; i++){
		for(j=1; j <= trueValue.n; j++){
			for(k=1; k <= trueValue.p; k++){
				diffs.values[i][j][k] = guess.values[i][j][k] - trueValue.values[i][j][k] ;
			}
		}
	}

	double normTrue = normGrid(trueValue);
	double normDiffs = normGrid(diffs);

	// if true norm is equal, then difference must be exactly zero.
	// else return NaN
	if(normTrue == 0.0 ){
		if(normDiffs == 0.0)
			return 0.0;
		else
			return 0.0/0.0;
	}


	return normDiffs / normTrue;
}


double maxDiffGrid(grid guess, grid trueValue){
	/*
	 Computes absolute value of maximum difference between two grids.
	 Ignores ghost zones.

	 Input:
		grid guess			Guess grid.
		grid *trueValue		True grid.

	 Output:
		double (returned)		Maximum difference between the two grids
	 */


	int i,j,k;

	double maxDiff = 0.0 ;
	double currentDiff ;

	for(i=1; i <= trueValue.m; i++){
		for(j=1; j <= trueValue.n; j++){
			for(k=1; k <= trueValue.p; k++){
				currentDiff = fabs(guess.values[i][j][k] - trueValue.values[i][j][k]) ;
				if(currentDiff > maxDiff)
					maxDiff = currentDiff ;
			}
		}
	}

	return maxDiff ;
}

vector normVectorField(vectorField v){
	/*
	 Calculates element-wise l2 norm of vector field.
	 Does not include ghost zones in norm calculation, but assumes they are included with the grid.
	 Calculates each component separately and returns answer as vector.

	 Input:
		vectorField g			Vector field to calculate norm.

	 Output:
	 vector (returned)  Element-wise l2 norm of vector field.
	 */


	vector runningNorm ;
	runningNorm.x = 0.0 ;
	runningNorm.y = 0.0 ;
	runningNorm.z = 0.0 ;

	int i,j,k;

	for(i=1; i <= v.m; i++){
		for(j=1; j <= v.n; j++){
			for(k=1; k <= v.p; k++){
				runningNorm.x += v.values[i][j][k].x * v.values[i][j][k].x ;
				runningNorm.y += v.values[i][j][k].y * v.values[i][j][k].y ;
				runningNorm.z += v.values[i][j][k].z * v.values[i][j][k].z ;
			}
		}
	}

	runningNorm.x = sqrt(runningNorm.x) ;
	runningNorm.y = sqrt(runningNorm.y) ;
	runningNorm.z = sqrt(runningNorm.z) ;

	return runningNorm;
}


vector l2RelativeErrVectorField(vectorField guess, vectorField trueValue){
	/*
	 Computes element-wise l2 relative error between two vector fields.
	 norm( guess - trueValue ) / norm(trueValue)

	 Input:
		vectorField guess			Guess grid.
		vectorField trueValue		True grid.

	 Output:
		double (returned)		l2 relative error between two input vectors.
	 */


	int i,j,k;

	vectorField diffs = initZeroVectorField3d(trueValue.m, trueValue.n, trueValue.p, trueValue.height) ;

	for(i=1; i <= trueValue.m; i++){
		for(j=1; j <= trueValue.n; j++){
			for(k=1; k <= trueValue.p; k++){
				diffs.values[i][j][k].x = guess.values[i][j][k].x - trueValue.values[i][j][k].x ;
				diffs.values[i][j][k].y = guess.values[i][j][k].y - trueValue.values[i][j][k].y ;
				diffs.values[i][j][k].z = guess.values[i][j][k].z - trueValue.values[i][j][k].z ;
			}
		}
	}

	vector normTrue = normVectorField(trueValue);
	vector normDiffs = normVectorField(diffs);
	vector relErr ;

	// if true norm is equal, then difference must be exactly zero.
	// else return NaN
	if(normTrue.x == 0.0 ){
		if(normDiffs.x == 0.0)
			relErr.x = 0.0;
		else
			relErr.x = 0.0/0.0;
	}
	else
		relErr.x = normDiffs.x / normTrue.x ;

	if(normTrue.y == 0.0 ){
		if(normDiffs.y == 0.0)
			relErr.y = 0.0;
		else
			relErr.y = 0.0/0.0;
	}
	else
		relErr.y = normDiffs.y / normTrue.y ;

	if(normTrue.z == 0.0 ){
		if(normDiffs.z == 0.0)
			relErr.z = 0.0;
		else
			relErr.z = 0.0/0.0;
	}
	else
		relErr.z = normDiffs.z / normTrue.z ;

	return relErr;
}


vector maxDiffVectorField(vectorField guess, vectorField trueValue){
	/*
	 Computes absolute value of maximum difference between two vector fields componentwise.
	 Ignores ghost zones.

	 Input:
		grid guess			Guess grid.
		grid *trueValue		True grid.

	 Output:
		vector maxDiff (returned)		Maximum difference between the two vector fields
	 */


	int i,j,k;

	vector maxDiff ;
	maxDiff.x = 0.0 ;
	maxDiff.y = 0.0 ;
	maxDiff.z = 0.0 ;

	vector currentDiff ;

	for(i=1; i <= trueValue.m; i++){
		for(j=1; j <= trueValue.n; j++){
			for(k=1; k <= trueValue.p; k++){

				currentDiff.x = fabs(guess.values[i][j][k].x - trueValue.values[i][j][k].x) ;
				if(currentDiff.x > maxDiff.x)
					maxDiff.x = currentDiff.x ;

				currentDiff.y = fabs(guess.values[i][j][k].y - trueValue.values[i][j][k].y) ;
				if(currentDiff.y > maxDiff.y)
					maxDiff.y = currentDiff.y ;

				currentDiff.z = fabs(guess.values[i][j][k].z - trueValue.values[i][j][k].z) ;
				if(currentDiff.z > maxDiff.z)
					maxDiff.z = currentDiff.z ;
			}
		}
	}

	return maxDiff ;
}


void printGrid(grid g){
	/*
	 Outputs contents of grid to stdout
	 For debugging use.

	 Input:
	 grid g   The grid to print.
	 */

	int i,j,k;

	for(k=0; k<g.p+2; k++){
		for(j=0; j<g.n+2; j++){
			for(i=0; i<g.m+2; i++){
				printf("%e ", g.values[i][j][k]) ;
			}
			printf("\n") ;
		}
		printf("\n\n") ;
	}

}

void multiplyGridByConstant(grid *g, double alpha){
	/*
	 Multiplies grid by a constant. Performed in place.
	 g = alpha * g ;
	 Does not multiply ghost zones.

	 Input:
	 grid *g    Address of grid to multiply
	 double alpha  Constant to multiply by

	 Output
	 grid *g    Modified grid
	 */

	int i,j,k;

	for(i=1; i <= g->m; i++){
		for(j=1; j <= g->n; j++){
			for(k=1; k <= g->p; k++){
				g->values[i][j][k] *= alpha ;
			}
		}
	}
}


void addGrids(grid first, grid second, grid *output){
	/*
	 Adds two grids not including ghost zones.
	 Stores result in output array, which may be identical to either input array

	 Input:
	 grid first      First grid to add
	 grid second     Second grid to add
	 grid *output    Result grid

	 Output
	 grid *output   Modified grid
	 */

	int i,j,k;

	for(i=1; i <= first.m; i++){
		for(j=1; j <= first.n; j++){
			for(k=1; k <= first.p; k++){
				output->values[i][j][k] = first.values[i][j][k] + second.values[i][j][k] ;
			}
		}
	}
}

void copyGridValues(grid g, grid *output){
	/*
	 Copys values from one grid to another including ghost zones.

	 Input:
	 grid g     Grid to copy
	 grid *output    Result grid

	 Output
	 grid *output   Modified grid
	 */

	int i,j,k;

	for(i=0; i <= g.m+1; i++){
		for(j=0; j <= g.n+1; j++){
			for(k=0; k <= g.p+1; k++){
				output->values[i][j][k] = g.values[i][j][k] ;
			}
		}
	}
}

void gridPlusConstantTimesGrid(grid first, double alpha, grid second, grid *output){
	/*
	 Adds two grids not including ghost zones.
	 output = first + alpha * second
	 Stores result in output array, which may be identical to either input array

	 Input:
	 grid first      First grid to add.
	 double alpha    Coefficient of second grid.
	 grid second     Second grid to add.
	 grid *output    Result grid.

	 Output:
	 grid *output   Modified grid.
	 */

	int i,j,k;

	for(i=1; i <= first.m; i++){
		for(j=1; j <= first.n; j++){
			for(k=1; k <= first.p; k++){
				output->values[i][j][k] = first.values[i][j][k] + alpha * second.values[i][j][k] ;
			}
		}
	}
}


void subtractGrids(grid first, grid second, grid *output){
	/*
	 Subtract two grids not including ghost zones.
	 output = first - second
	 Stores result in output array, which may be identical to either input array

	 Input:
	 grid first      First grid to add
	 grid second     Second grid to add
	 grid *output    Result grid

	 Output
	 grid *output   Modified grid
	 */

	int i,j,k;

	for(i=1; i <= first.m; i++){
		for(j=1; j <= first.n; j++){
			for(k=1; k <= first.p; k++){
				output->values[i][j][k] = first.values[i][j][k] - second.values[i][j][k] ;
			}
		}
	}
}

double innerProductGrids(grid first, grid second){
	/*
	 Calculates element wise inner product of two grids not including ghost zones.

	 Input:
	 grid first      First grid to inner product.
	 grid second     Second grid to inner product.

	 Output
	 double value (returned) The element wise inner product of the grids.
	 */

	int i,j,k;
	double value = 0.0;

	for(i=1; i <= first.m; i++){
		for(j=1; j <= first.n; j++){
			for(k=1; k <= first.p; k++){
				value += first.values[i][j][k] * second.values[i][j][k] ;
			}
		}
	}

	return value;
}


void printVectorField(vectorField v){
	/*
	 Outputs contents of vector field to stdout
	 For debugging use.

	 Input:
	 vectorField v   The grid to print.
	 */

	int i,j,k;

	printf("x components:\n") ;
	for(k=0; k<v.p+2; k++){
		for(j=0; j<v.n+2; j++){
			for(i=0; i<v.m+2; i++){
				printf("%e ", v.values[i][j][k].x) ;
			}
			printf("\n") ;
		}
		printf("\n\n") ;
	}

	printf("y components:\n") ;
	for(k=0; k<v.p+2; k++){
		for(j=0; j<v.n+2; j++){
			for(i=0; i<v.m+2; i++){
				printf("%e ", v.values[i][j][k].y) ;
			}
			printf("\n") ;
		}
		printf("\n\n") ;
	}

	printf("z components:\n") ;
	for(k=0; k<v.p+2; k++){
		for(j=0; j<v.n+2; j++){
			for(i=0; i<v.m+2; i++){
				printf("%e ", v.values[i][j][k].z) ;
			}
			printf("\n") ;
		}
		printf("\n\n") ;
	}
}










