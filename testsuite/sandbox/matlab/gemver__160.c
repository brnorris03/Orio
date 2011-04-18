void
GEMVER (float *A, int A_nrows, int A_ncols, float a, float b, float *u1,
		int u1_nrows, float *u2, int u2_nrows, float *v1, int v1_nrows,
		float *v2, int v2_nrows, float *y, int y_nrows, float *z, int z_nrows,
		float *B, int B_nrows, int B_ncols, float *w, int w_nrows, float *x,
		int x_nrows)
{
	float *t14 = new float[A_ncols];
	float *t15 = new float[A_ncols];
	float *t19 = new float[A_nrows];

	for (int i = 0; i < A_ncols; i += 1) {
		float t31 = v2[i];
		float t24 = v1[i];
		float *t27 = A + i * A_nrows;
		float *t37 = B + i * A_nrows;

		for (int j = 0; j < A_nrows; j += 1) {
			float t61 = u2[j];
			float t54 = u1[j];
			float t57 = t27[j];
			float &t67 = t37[j];

			t67 = ((t57 + (t54 * t24)) + (t61 * t31));
		}
	}
	for (int i = 0; i < A_ncols; i += 1) {
		float *t38 = B + i * A_nrows;
		float &t40 = t14[i];
		float t70 = 0;

		for (int j = 0; j < A_nrows; j += 1) {
			float t69 = t38[j];
			float t68 = y[j];

			t70 += (t68 * t69);
		}
		t40 = t70;
	}
	for (int i = 0; i < A_ncols; i += 1) {
		float t41 = t14[i];
		float &t43 = t15[i];

		t43 = (b * t41);
	}
	for (int i = 0; i < A_ncols; i += 1) {
		float t45 = z[i];
		float t44 = t15[i];
		float &t47 = x[i];

		t47 = (t44 + t45);
	}
	for (int ii = 0; ii < A_nrows; ++ii)
		t19[ii] = 0.0;
	for (int i = 0; i < A_ncols; i += 1) {
		float t49 = x[i];
		float *t48 = B + i * A_nrows;

		for (int j = 0; j < A_nrows; j += 1) {
			float t71 = t48[j];
			float &t73 = t19[j];

			t73 += (t49 * t71);
		}
	}
	for (int i = 0; i < A_nrows; i += 1) {
		float t51 = t19[i];
		float &t53 = w[i];

		t53 = (a * t51);
	}
}
