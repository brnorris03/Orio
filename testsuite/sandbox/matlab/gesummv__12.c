void
GESUMMV (float *A, int A_nrows, int A_ncols, float *B, int B_nrows,
		 int B_ncols, float a, float b, float *x, int x_nrows, float *y,
		 int y_nrows)
{
	float *t7 = new float[B_nrows];
	float *t3 = new float[A_nrows];

	for (int ii = 0; ii < A_nrows; ++ii)
		t3[ii] = 0.0;
	for (int ii = 0; ii < B_nrows; ++ii)
		t7[ii] = 0.0;
	for (int i = 0; i < A_ncols; i += 1) {
		float *t17 = B + i * B_nrows;
		float t12 = x[i];
		float *t11 = A + i * A_nrows;

		for (int j = 0; j < A_nrows; j += 1) {
			float t30 = t17[j];
			float &t32 = t7[j];
			float t27 = t11[j];
			float &t29 = t3[j];

			t32 += (t30 * t12);
			t29 += (t12 * t27);
		}
	}
	for (int i = 0; i < A_nrows; i += 1) {
		float t20 = t7[i];
		float t14 = t3[i];
		float &t26 = y[i];

		t26 = ((a * t14) + (b * t20));
	}
}
