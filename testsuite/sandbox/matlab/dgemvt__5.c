void
DGEMVT (float *A, int A_nrows, int A_ncols, float a, float b, float *y,
		int y_nrows, float *z, int z_nrows, float *w, int w_nrows, float *x,
		int x_nrows)
{
	float *t5 = new float[A_ncols];
	float *t9 = new float[A_nrows];

	for (int i = 0; i < A_ncols; i += 1) {
		float *t13 = A + i * A_nrows;
		float &t18 = t5[i];
		float t31 = 0;

		for (int j = 0; j < A_nrows; j += 1) {
			float t30 = t13[j];
			float t29 = y[j];

			t31 += (t29 * t30);
		}
		t18 = (b * t31);
	}
	for (int i = 0; i < A_ncols; i += 1) {
		float t20 = z[i];
		float t19 = t5[i];
		float &t22 = x[i];

		t22 = (t19 + t20);
	}
	for (int ii = 0; ii < A_nrows; ++ii)
		t9[ii] = 0.0;
	for (int i = 0; i < A_ncols; i += 1) {
		float t24 = x[i];
		float *t23 = A + i * A_nrows;

		for (int j = 0; j < A_nrows; j += 1) {
			float t32 = t23[j];
			float &t34 = t9[j];

			t34 += (t24 * t32);
		}
	}
	for (int i = 0; i < A_nrows; i += 1) {
		float t26 = t9[i];
		float &t28 = w[i];

		t28 = (a * t26);
	}
}
