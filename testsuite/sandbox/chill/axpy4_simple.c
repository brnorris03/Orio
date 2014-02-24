void axpy4(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4) {

/*@ begin CHiLL (
 
  	transform Recipe(recipe.lua)
  ) @*/

int i;
for (i=0; i<N; i++)
  y[i] = y[i] + a1*x1[i] + a2*x2[i] + a3*x3[i] + a4*x4[i];
/*@ end @*/   // CHiLL


}
