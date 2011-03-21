
#include "gridUtil.h"
#include "differentialOps.h"

/*
 Differential operators for structured grids.

 Alex Kaiser, LBNL, 7/2010
*/


void centralDifference3D(grid g, grid *dx, grid *dy, grid *dz){
	/*
	 Numerically approximates the first partial derivatives of the 3D grid g in each direction.
	 Ghost zones must be included with input.
	 Ghost zones are not modified in output arrays.

	 Input:
	 grid g  The grid of which to approximate partials. Ghost zones included
	 grid *dx  Address of preallocated, zeroed grid.
	 grid *dy  Address of preallocated, zeroed grid.
	 grid *dz  Address of preallocated, zeroed grid.
	 double height Grid height

	 Output:
	 grid dx  Approximation to partial derivative in x direction.
	 grid dy  Approximation to partial derivative in y direction.
	 grid dz  Approximation to partial derivative in z direction.
	 */

	int i,j,k;
	double coeff = 1.0 / (2.0 * g.height) ;

	for(i=1; i <= g.m; i++){
		for(j=1; j <= g.n; j++){
			for(k=1; k <= g.p; k++){
				 dx->values[i][j][k] = coeff * (g.values[i+1][j][k] - g.values[i-1][j][k]) ;
				 dy->values[i][j][k] = coeff * (g.values[i][j+1][k] - g.values[i][j-1][k]) ;
				 dz->values[i][j][k] = coeff * (g.values[i][j][k+1] - g.values[i][j][k-1]) ;
			}
		}
	}

}


void divergence(vectorField v, grid *div){
	/*
	 Numerically approximates the divergence of the 3D grid g in each direction.
	 Ghost zones must be included with input.
	 Ghost zones are not modified in output arrays.

	 Input:
	 vectorField v  The vector field of which to approximate divergence. Ghost zones included
	 grid *div      Address of preallocated, zeroed grid

	 Output:
	 grid div   The divergence of the input grid
	 */

	int i,j,k;
	double coeff = 1.0 / (2.0 * v.height) ;

	for(i=1; i <= v.m; i++){
		for(j=1; j <= v.n; j++){
			for(k=1; k <= v.p; k++){
				 div->values[i][j][k] = coeff * ( (v.values[i+1][j][k].x - v.values[i-1][j][k].x) +
				   				   				  (v.values[i][j+1][k].y - v.values[i][j-1][k].y) +
				  				   				  (v.values[i][j][k+1].z - v.values[i][j][k-1].z) );
			}
		}
	}

}

void curl(vectorField v, vectorField *curl){
	/*
	 Numerically approximates the curl of the 3D vector field v.
	 Ghost zones must be included with input.
	 Ghost zones are not modified in output arrays.

	 Input:
	 vectorField v  The vector field of which to approximate curl. Ghost zones included.
	 vectorField *curl    Address of preallocated, zeroed vector field.

	 Output:
	 vectorField *curl   Curl of the supplied grid.
	  */

	int i,j,k;
	double coeff = 1.0 / (2.0 * v.height) ;

	for(i=1; i <= v.m; i++){
		for(j=1; j <= v.n; j++){
			for(k=1; k <= v.p; k++){
				 curl->values[i][j][k].x = coeff * ( (v.values[i][j+1][k].z - v.values[i][j-1][k].z) -
				   				   				     (v.values[i][j][k+1].y - v.values[i][j][k-1].y) );

				 curl->values[i][j][k].y = coeff * ( (v.values[i][j][k+1].x - v.values[i][j][k-1].x) -
				   				   				     (v.values[i+1][j][k].z - v.values[i-1][j][k].z) );

				 curl->values[i][j][k].z = coeff * ( (v.values[i+1][j][k].y - v.values[i-1][j][k].y) -
				   				   				     (v.values[i][j+1][k].x - v.values[i][j-1][k].x) );
			}
		}
	}

}


void gradient(grid g, vectorField *gradient){
	/*
	 Numerically approximates the gradient of the grid g.
	 Ghost zones must be included with input.
	 Ghost zones are not modified in output arrays.

	 Input:
	 grid g  The vector field of which to approximate gradient. Ghost zones included.
	 vectorField *gradient    Address of preallocated, zeroed vector field.

	 Output:
	 vectorField *gradient   Gradient of the supplied grid.
	  */

	int i,j,k;
	double coeff = 1.0 / (2.0 * g.height) ;

	for(i=1; i <= g.m; i++){
		for(j=1; j <= g.n; j++){
			for(k=1; k <= g.p; k++){
				 gradient->values[i][j][k].x = coeff * (g.values[i+1][j][k] - g.values[i-1][j][k]) ;
				 gradient->values[i][j][k].y = coeff * (g.values[i][j+1][k] - g.values[i][j-1][k]) ;
				 gradient->values[i][j][k].z = coeff * (g.values[i][j][k+1] - g.values[i][j][k-1]) ;
			}
		}
	}

}



void laplacian(grid g, grid *laplacian){
	/*
	 Numerically approximates the Laplacian of the grid g.
	 Ghost zones must be included with input.
	 Ghost zones are not modified in output arrays.

	 Input:
	 grid g  The vector field of which to approximate Laplacian. Ghost zones included.
	 grid *laplacian      Address of preallocated, zeroed grid

	 Output:
	 grid laplacian   The Laplacian of the input grid
	 */

	int i,j,k;
	double alpha = -6.0 / (g.height * g.height) ;
	double beta = 1.0 / (g.height * g.height) ;

	for(i=1; i <= g.m; i++){
		for(j=1; j <= g.n; j++){
			for(k=1; k <= g.p; k++){
				 laplacian->values[i][j][k] = alpha * g.values[i][j][k] +
											 beta * ( (g.values[i+1][j][k] + g.values[i-1][j][k]) +
													  (g.values[i][j+1][k] + g.values[i][j-1][k]) +
													  (g.values[i][j][k+1] + g.values[i][j][k-1]) ) ;
			}
		}
	}
}





