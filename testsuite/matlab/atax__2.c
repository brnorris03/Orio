void
ATAX (float *A, int A_nrows, int A_ncols, float *x, int x_nrows, float *y,
	  int y_nrows)
{
	for (int ii = 0; ii < y_nrows; ++ii)
		y[ii] = 0.0;
	for (int i = 0; i < A_nrows; i += 1) {
		float *t6 = A + i * A_ncols;
		float t14 = 0;

		for (int j = 0; j < A_ncols; j += 1) {
			float t13 = x[j];
			float t12 = t6[j];

			t14 += (t12 * t13);
		}
		for (int j = 0; j < A_ncols; j += 1) {
			float t15 = t6[j];
			float &t17 = y[j];

			t17 += (t15 * t14);
		}
	}
}
