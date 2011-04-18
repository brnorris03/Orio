void
AATX (float *A, int A_nrows, int A_ncols, float *x, int x_nrows, float *y,
	  int y_nrows)
{
	for (int ii = 0; ii < y_nrows; ++ii)
		y[ii] = 0.0;
	for (int i = 0; i < A_ncols; i += 1) {
		float *t9 = A + i * A_nrows;
		float t14 = 0;

		for (int j = 0; j < A_nrows; j += 1) {
			float t13 = t9[j];
			float t12 = x[j];

			t14 += (t12 * t13);
		}
		for (int j = 0; j < A_nrows; j += 1) {
			float t15 = t9[j];
			float &t17 = y[j];

			t17 += (t15 * t14);
		}
	}
}
