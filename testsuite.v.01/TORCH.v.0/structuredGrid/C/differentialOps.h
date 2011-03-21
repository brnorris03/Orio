

/*
 Header for differential operators for structured grids

 Alex Kaiser, LBNL, 7/2010
*/


void centralDifference3D(grid g, grid *dx, grid *dy, grid *dz) ;

void divergence(vectorField v, grid *div) ;

void curl(vectorField v, vectorField *curl) ;

void gradient(grid g, vectorField *gradient) ;

void laplacian(grid g, grid *laplacian) ;
