#include <stdlib.h>
void DGEMV(int A_nrows, int A_ncols, double A[A_ncols][A_nrows], int e_nrows, double e[e_nrows], int w_nrows, double w[w_nrows], int x_nrows, double x[x_nrows], int p_nrows, double p[p_nrows], int y_nrows, double y[y_nrows], int z_nrows, double z[z_nrows]){
	int ii,i,j;
	double t2[A_nrows];
	for (ii = 0; ii < A_nrows; ++ii)
		t2[ii] = 0.0; //in translate_tmp
	double t6[A_nrows];
	for (ii = 0; ii < A_nrows; ++ii)
		t6[ii] = 0.0; //in translate_tmp
	double t10[A_nrows];
	for (ii = 0; ii < A_nrows; ++ii)
		t10[ii] = 0.0; //in translate_tmp
	for (i = 0;i < A_ncols; i+=1) {
		// 1
		for (j = 0;j < A_nrows; j+=1) {
			// 7
			t10[j] += (e[i]*A[i][j]); //accessing indexMap[10]
			t6[j] += (w[i]*A[i][j]); //accessing indexMap[6]
			t2[j] += (A[i][j]*x[i]); //accessing indexMap[2]
		}
	}
	for (i = 0;i < A_nrows; i+=1) {
		// 2
		y[i] = (t2[i]+y[i]); //accessing indexMap[14]
	}
	for (i = 0;i < A_nrows; i+=1) {
		// 4
		z[i] = (t6[i]+z[i]); //accessing indexMap[15]
	}
	for (i = 0;i < A_nrows; i+=1) {
		// 6
		p[i] = (t10[i]+p[i]); //accessing indexMap[13]
	}
}
