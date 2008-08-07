void
AxATy (float *A, int A_nrows, int A_ncols, float *x, int x_nrows, float *y,
	   int y_nrows, float *w, int w_nrows, float *z, int z_nrows)
{
	for (int ii = 0; ii < z_nrows; ++ii)
		z[ii] = 0.0;
	for (int i = 0; i < A_nrows; i += 1) {
		float t12 = y[i];
		float *t8 = A + i * A_ncols;
		float &t10 = w[i];
		float t16 = 0;

		for (int j = 0; j < A_ncols; j += 1) {
			float t15 = x[j];
			float t14 = t8[j];
			float &t19 = z[j];

			t19 += (t12 * t14);
			t16 += (t14 * t15);
		}
		t10 = t16;
	}
}
