
#include <stdio.h>
#include <math.h>
#include "gridUtil.h"
#include "testFunctions.h"


/*
 Test functions for structured grid kernels and tests.

 Alex Kaiser, LBNL, 7/2010
 */

double centralDiffsTestFn(double x, double y, double z, int type, double toughness){
	/*
	  Test function for central differences.

	  Input:
	  double x,y,z         The point at which to evaluate the function.
	  int type             Type number for function to return.
	  double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  double (returned) The function value at that point.
	 */


	if(type == 0)
		return sin(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z);

	if(type == 1)
		return toughness*x*y*z + pow(toughness*x*y*z, 2) + pow(toughness*x*y*z, 3) ;

	if(type == 2)
		return exp(toughness*x*y*z) ;

	fprintf(stderr, "Unsupported type. Using default type.\n");

	return sin(x) * cos(y) * pow(z,3) ;
}

double centralDiffsDxSoln(double x, double y, double z, int type, double toughness){
	/*
	  Solution for function output of central differences.

	  Input:
	 double x,y,z  The point at which to evaluate Dx.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  double (returned) The value of Dx at that point.
	 */

	if(type == 0)
		return toughness * cos(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z);

	if(type == 1)
		return toughness*y*z + 2*x*pow(toughness*y*z, 2) + 3*x*x*pow(toughness*y*z, 3) ;

	if(type == 2)
		return toughness*y*z * exp(toughness*x*y*z) ;

	return cos(x) * cos(y) * pow(z,3) ;
}


double centralDiffsDySoln(double x, double y, double z, int type, double toughness){
	/*
	  Solution for function output of central differences.

	  Input:
	 double x,y,z  The point at which to evaluate Dy.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  double (returned) The value of Dx at that point.
	 */

	if(type == 0)
		return  sin(toughness * x) * 2.0 * toughness * cos(2.0 * toughness * y) * sin(4.0 * toughness * z);

	if(type == 1)
		return toughness*x*z + 2*y*pow(toughness*x*z, 2) + 3*y*y*pow(toughness*x*z, 3) ;

	if(type == 2)
		return toughness*x*z * exp(toughness*x*y*z) ;

	return sin(x) * -sin(y) * pow(z,3) ;
}


double centralDiffsDzSoln(double x, double y, double z, int type, double toughness){
	/*
	  Solution for function output of central differences.

	  Input
	 double x,y,z  The point at which to evaluate Dz.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output
	  double (returned) The value of Dx at that point.
	 */

	if(type == 0)
		return sin(toughness * x) * sin(2.0 * toughness * y) * 4.0 * toughness * cos(4.0 * toughness * z);

	if(type == 1)
		return toughness*x*y + 2*z*pow(toughness*x*y, 2) + 3*z*z*pow(toughness*x*y, 3) ;

	if(type == 2)
		return toughness*x*y * exp(toughness*x*y*z) ;

	return sin(x) * cos(y) * 3 * z * z;
}


vector divergenceTestFn(double x, double y, double z, int type, double toughness){
	/*
	  Test function for divergence. Use to initialize a vector field.

	  Input:
	  double x,y,z  The point at which to evaluate the function.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  vector (returned) The function vector value at that point.
	 */

	if(type == 0){
		vector v;
		v.x = sin(toughness * x) ;
		v.y = sin(2.0 * toughness * y) ;
		v.z = sin(4.0 * toughness * z) ;
		return v;
	}
	if(type == 1){ // divergence free
		vector v;
		v.x = toughness * x * y ;
		v.y = toughness * x * y * z ;
		v.z = - (toughness * y * z + 0.5 * toughness * x * z*z) ;
		return v;
	}
	if(type == 2){
		vector v;
		v.x = exp(toughness*x) ;
		v.y = exp(2.0*toughness*y) ;
		v.z = exp(4.0*toughness*z) ;
		return v;
	}

	fprintf(stderr, "Unsupported type. Using default type.\n");

	vector v ;
	v.x = sin(x) * cos(y) ;
	v.y = sin(y) * pow(z, 3) ;
	v.z = cos(x) * cos(y) * z ;
	return v;
}

double divergenceSoln(double x, double y, double z, int type, double toughness){
	/*
	  Analytic solution to divergence specified above.

	  Input:
	 double x,y,z  The point at which to evaluate divergence.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  double (returned) The value of divergence at that point.
	 */


	if(type == 0){
		return toughness * cos(toughness * x) +
				2.0 * toughness * cos(2.0 * toughness * y) +
				4.0 * toughness * cos(4.0 * toughness * z);
	}

	if(type == 1)
		return 0.0 ;

	if(type == 2)
		return toughness * (exp(toughness * x) + 2.0*exp(2.0*toughness*y) + 4.0*exp(4.0*toughness*z) ) ;

	return cos(x)*cos(y) + cos(y)*pow(z,3) + cos(x)*cos(y) ;
}


vector curlTestFn(double x, double y, double z, int type, double toughness){
	/*
	  Test function for curl. Use to initialize a vector field.

	  Input:
	 double x,y,z  The point at which to evaluate the function.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  vector (returned) The function vector value at that point.
	 */

	if(type == 0){
		vector v;
		v.x = sin(toughness * z) ;
		v.y = sin(2.0 * toughness * x) ;
		v.z = sin(4.0 * toughness * y) ;
		return v;
	}

	if(type == 1){
		vector v;
		v.x = toughness * x * x * y * z ;
		v.y = toughness * (z * pow(x,3)) / 3.0 ;
		v.z = toughness * (y * pow(x,3)) / 3.0 ;
		return v;
	}

	if(type == 2){
		vector v;
		v.x = exp(toughness * z) ;
		v.y = exp(2.0 * toughness * x) ;
		v.z = exp(4.0 * toughness * y) ;
		return v;
	}


	fprintf(stderr, "Unsupported type. Using default type.\n");

	vector v ;
	v.x = sin(y) ;
	v.y = cos(z) ;
	v.z = 5 * x * y ;
	return v;
}

vector curlSoln(double x, double y, double z, int type, double toughness){
	/*
	  Analytic solution to curl specified at above.

	  Input:
	 double x,y,z  The point at which to evaluate curl.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  vector The value of curl at that point.
	 */

	if(type == 0){
		vector v;
		v.x = 4.0 * toughness * cos(4.0 * toughness * y) ;
		v.y =       toughness * cos(      toughness * z) ;
		v.z = 2.0 * toughness * cos(2.0 * toughness * x) ;
		return v;
	}

	if(type == 1){
		// conservative field
		vector v = {0.0, 0.0, 0.0};
		return v;
	}

	if(type == 2){
		vector v;
		v.x = 4.0 * toughness * exp(4.0 * toughness * y) ;
		v.y =       toughness * exp(      toughness * z) ;
		v.z = 2.0 * toughness * exp(2.0 * toughness * x) ;
		return v;
	}

	vector v;
	v.x = 5*x + sin(z) ;
	v.y = -5 * y ;
	v.z = -cos(y) ;
	return v ;
}

double gradientTestFn(double x, double y, double z, int type, double toughness){
	/*
	  Test function for gradient.

	  Input:
	 double x,y,z  The point at which to evaluate the function.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  double (returned) The function value at that point.
	 */

	if(type == 0)
		return sin(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z);

	if(type == 1)
		return toughness*x*y*z + pow(toughness*x*y*z, 2) + pow(toughness*x*y*z, 3) ;

	if(type == 2)
		return exp(toughness*x*y*z) ;

	fprintf(stderr, "Unsupported type. Using default type.\n");

	return sin(x) * cos(y) * pow(z,3) ;
}

vector gradientSoln(double x, double y, double z, int type, double toughness){
	/*
	  Analytic solution to gradient specified at above.

	  Input:
	 double x,y,z  The point at which to evaluate gradient.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  vector The value of gradient at that point.
	 */

	if(type == 0){
		vector v;
		v.x = toughness * cos(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z);
		v.y = sin(toughness * x) * 2.0 * toughness * cos(2.0 * toughness * y) * sin(4.0 * toughness * z);
		v.z = sin(toughness * x) * sin(2.0 * toughness * y) * 4.0 * toughness * cos(4.0 * toughness * z);
		return v;
	}
	if(type == 1){
		vector v;
		v.x = toughness*y*z + 2*x*pow(toughness*y*z, 2) + 3*x*x*pow(toughness*y*z, 3) ;
		v.y = toughness*x*z + 2*y*pow(toughness*x*z, 2) + 3*y*y*pow(toughness*x*z, 3) ;
		v.z = toughness*x*y + 2*z*pow(toughness*x*y, 2) + 3*z*z*pow(toughness*x*y, 3) ;
		return v;
	}
	if(type == 2){
		vector v;
		v.x = toughness*y*z * exp(toughness*x*y*z) ;
		v.y = toughness*x*z * exp(toughness*x*y*z) ;
		v.z = toughness*x*y * exp(toughness*x*y*z) ;
		return v;
	}

	vector v ;
	v.x = cos(x) * cos(y) * pow(z,3) ;
	v.y = -sin(x) * sin(y) * pow(z,3) ;
	v.z = sin(x) * cos(y) * 3 * z * z ;
	return v ;
}

double laplacianTestFn(double x, double y, double z, int type, double toughness){
	/*
	  Test function for Laplacian.

	  Input:
	 double x,y,z  The point at which to evaluate the function.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  double (returned) The function value at that point.
	 */


	if(type == 0)
		return sin(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z);

	if(type == 1)
		return toughness*x*y*z + pow(toughness*x*y*z, 2) + pow(toughness*x*y*z, 3) ;

	if(type == 2)
		return exp(toughness*x*y*z) ;

	fprintf(stderr, "Unsupported type. Using default type.\n");

	return sin(x) * cos(y) * pow(z,3) ;
}


double laplacianSoln(double x, double y, double z, int type, double toughness){
	/*
	  Analytic solution to Laplacian specified above.

	  Input:
	 double x,y,z  The point at which to evaluate Laplacian.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  double (returned) The value of Laplacian at that point.
	 */

	if(type == 0)
		return (-21.0 * toughness * toughness) * (sin(toughness * x) * sin(2.0 * toughness * y) * sin(4.0 * toughness * z)) ;

	if(type == 1)
		return (2.0*toughness*toughness) * (1.0 + 3.0*toughness*x*y*z) * (x*x*y*y + x*x*z*z + y*y*z*z) ;

	if(type == 2)
		return (toughness * toughness * (x*x*y*y + x*x*z*z + y*y*z*z)) * exp(toughness*x*y*z) ;

	return -2 * sin(x)*cos(y)*pow(z,3) + 6*sin(x)*cos(y)*z ;
}

double laplacianTestFnHomogeneous(double x, double y, double z, int type, double toughness){
	/*
	  Test function for Homogeneous Laplacian.
	  Returns functions with zero value on boundary.

	  Input:
	 double x,y,z  The point at which to evaluate the function.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  double (returned) The function value at that point.
	 */

	// select integer toughness to ensure correct boundary conditions.
	toughness = round(toughness) ;
	if(toughness < 1)
		toughness = 1.0;

	double pi = acos(-1);

	if(type == 0)
		return 0.025 * sin(toughness * pi * x) * sin(pi * toughness * y) * sin(pi * toughness * z);

	if(type == 1)
		return toughness * (x*x - x) * (y*y - y) * (z*z - z) ;

	fprintf(stderr, "Unsupported type. Using default type.\n");

	return 0.025 * sin(toughness * pi * x) * sin(pi * toughness * y) * sin(pi * toughness * z);
}


double laplacianSolnHomogeneous(double x, double y, double z, int type, double toughness){
	/*
	  Analytic solution to Laplacian specified above.
	  Returns functions with zero value on boundary.

	  Input:
	 double x,y,z  The point at which to evaluate Laplacian.
	 int type             Type number for function to return.
	 double toughness     Difficulty parameter. Larger values result in more numerically challenging computations.

	  Output:
	  double (returned) The value of Laplacian at that point.
	 */

	// select integer toughness to ensure correct boundary conditions.
	toughness = round(toughness) ;
	if(toughness < 1)
		toughness = 1.0;

	double pi = acos(-1);

	if(type == 0)
		return - (3.0 * toughness * toughness * pi * pi) *
		          0.025 * sin(toughness * pi * x) * sin(pi * toughness * y) * sin(pi * toughness * z);

	if(type == 1)
		return 2 * toughness * ((x*x - x)*(y*y - y) + (x*x - x)*(z*z - z) + (y*y - y)*(z*z - z)) ;

	return - (3 * toughness * toughness * pi * pi) *
             0.025 * sin(toughness * pi * x) * sin(pi * toughness * y) * sin(pi * toughness * z);
}


double heatEqnInitialConds(double x, double y, double z, int type, double toughness){
	/*
	 Returns initial conditions for heat equation solve.
	 Function is sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z).
	 Use to set up grid for initial conditions.

	 Input:
	 double x,y,z  The point at which to evaluate the initial conditions.
	 int type      Filler parameter. Should be removed but only easy way to do this would be operator overloading.
	 double toughness  Filler parameter. Should be removed but only easy way to do this would be operator overloading.

	 Output:
	 double (returned) The value of initial conditions at that point.
	 */

	double pi = acos(-1) ;
	return sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z);
}

double heatEqnAnalyticSoln(double x, double y, double z, double t, int type, double toughness){
	/*
	 Returns solution to heat eqn at given location at space and time.
	 Function is sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z).
	 Use to set up grid for initial conditions.

	 Input:
	 double x,y,z  The point at which to evaluate the initial conditions.
 	 int type      Filler parameter. Should be removed but only easy way to do this would be operator overloading.
	 double toughness  Filler parameter. Should be removed but only easy way to do this would be operator overloading.

	 Output:
	 double (returned) The value of initial conditions at that point.
	 */
	double pi = acos(-1) ;
	return exp(-12*pi*pi*t) * (sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z));
}


double plane(double x, double y, double z, int type, double toughness){
	/*
	 For debugging, returns a plane.

	 Input:
	 double x,y,z  The point at which to evaluate the initial conditions.
	 int type      Filler parameter. Should be removed but only easy way to do this would be operator overloading.
	 double toughness  Filler parameter. Should be removed but only easy way to do this would be operator overloading.

	 Output:
	 double (returned) The value of initial conditions at that point.
	 */
	return x + y + z;
}

