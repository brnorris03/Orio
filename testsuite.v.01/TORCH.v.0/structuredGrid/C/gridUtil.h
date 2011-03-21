/*
 Structured grid utilities header.

 Alex Kaiser, LNBL, 7/2010
 */


typedef struct {
	double x;
	double y;
	double z;
}vector;


typedef struct {
	double ***values ;
	int m, n, p; // Internal grid dimensions.
				 // Values allocated two larger in each dimension to include ghost zones.
	double height;
}grid;

typedef struct {
	grid *grids;
	int tSteps ;
	double dt;
}gridArray;

typedef struct {
	vector ***values ;
	int m, n, p;
	double height;
}vectorField;

typedef struct {
	grid guess;
	grid rhs;
	grid residual;
	double lambda;
}mgGrid;

typedef struct {
	mgGrid *mgGrids ;
	int depth ;
}mgGridArray;



// allocators and initialization

//use allocator from complexUtil.h because this produces a duplicate symbol
// double *** allocDouble3d(int m, int n, int p);
// void freeDouble3d(double ***z, int m, int n, int p);

grid initGrid3d( double f(double xArg, double yArg, double zArg, int type, double toughness), double height, double x, double y, double z, int type, double toughness);
void setGrid3d(grid *g, double f(double xArg, double yArg, double zArg, int typeArg, double toughnessArg), int type, double toughness);
grid initZeroGrid3d(int m, int n, int p, double height);
void setZeroGrid3d(grid *g);
void freeGrid(grid g) ;

vector *** allocVector3d(int m, int n, int p);
void freeVector3d(vector ***z, int m, int n, int p);

vectorField initVectorField3d( vector f(double xArg, double yArg, double zArg, int typeArg, double toughnessArg), double height, double x, double y, double z, int type, double toughness) ;
void setVectorField3d(vectorField *v, vector f(double xArg, double yArg, double zArg, int typeArg, double toughnessArg), int type, double toughness);
vectorField initZeroVectorField3d(int m, int n, int p, double height) ;
void setZeroVectorField3d(vectorField *v) ;
void freeVectorField(vectorField f) ;

void freeGridArray(gridArray currentArray);

mgGridArray allocMultiGridArrays(int m, int n, int p, double height, double lambda, int depth) ;

void gridToVector(grid g, double *x);
void vectorToGrid(grid *g, double *x);

// arithmetic
double normGrid(grid g);
double l2RelativeErrGrid(grid guess, grid trueValue);
double maxDiffGrid(grid guess, grid trueValue);

vector normVectorField(vectorField v) ;
vector l2RelativeErrVectorField(vectorField guess, vectorField trueValue);
vector maxDiffVectorField(vectorField guess, vectorField trueValue) ;

void multiplyGridByConstant(grid *g, double alpha);
void addGrids(grid first, grid second, grid *output);
void subtractGrids(grid first, grid second, grid *output);
void gridPlusConstantTimesGrid(grid first, double alpha, grid second, grid *output);
void copyGridValues(grid g, grid *output);
double innerProductGrids(grid first, grid second);

// I/O
void printGrid(grid g) ;
void printVectorField(vectorField v) ;


