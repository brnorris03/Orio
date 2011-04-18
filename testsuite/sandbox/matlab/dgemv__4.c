void
DGEMV (float *A, int A_nrows, int A_ncols, float alpha, float beta, float *x,
	   int x_nrows, float *y, int y_nrows, float *z, int z_nrows)
{
	float *t3 = new float[A_nrows];

	for (int ii = 0; ii < A_nrows; ++ii)
		t3[ii] = 0.0;
	for (int i = 0; i < A_ncols; i += 1) {
		float t11 = x[i];
		float *t10 = A + i * A_nrows;

		for (int j = 0; j < A_nrows; j += 1) {
			float t23 = t10[j];
			float &t25 = t3[j];

			t25 += (t11 * t23);
		}
	}
	for (int i = 0; i < A_nrows; i += 1) {
		float t16 = y[i];
		float t13 = t3[i];
		float &t22 = z[i];

		t22 = ((alpha * t13) + (beta * t16));
	}
}
